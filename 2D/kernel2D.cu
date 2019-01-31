/**
 *	Based off earlier start from:
 *	https://github.com/Robadob/SP-Bench/commit/35dcbb81cc0b73cdb6b08fb622f13e688a878133
 */
#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include <stdio.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>
#include <cub/cub.cuh>
#include <glm/gtc/epsilon.hpp>
#define EPSILON 0.0001f
//#define CIRCLES
//Cuda call
static void HandleCUDAError(const char *file,
	int line,
	cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
	cudaDeviceSynchronize();
#endif
	if (status != cudaError::cudaSuccess || (status = cudaGetLastError()) != cudaError::cudaSuccess)
	{
		printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
#ifdef _DEBUG
		getchar();
#endif
		exit(1);
	}
}
#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))

//Logging (found in log.cpp)
#include <fstream>
void createLog(std::ofstream &f);
void log(std::ofstream &f,
	const unsigned int &estRadialNeighbours,
	const unsigned int &agentCount,
	const unsigned int &envWidth,
	const float &PBM_control,
	const float &kernel_control,
	const float &PBM,
	const float &kernel,
	const unsigned int &fails
);
__device__ __constant__ unsigned int d_agentCount;
__device__ __constant__ float d_environmentWidth_float;
__device__ __constant__ unsigned int d_gridDim;
glm::uvec2 GRID_DIMS;
__device__ __constant__ float d_gridDim_float;
__device__ __constant__ float d_RADIUS;
__device__ __constant__ float d_R_SIN_45;
__device__ __constant__ float d_binWidth;

//For thread block max bin check
unsigned int *d_PBM_max_count;
unsigned int PBM_max_count = 0;
unsigned int PBM_max_Moore_count = 0;//This is unused, it could be used if we wished to load entire Moore neighbourhood at once to shared mem, instead we load a bin at a time

texture<float2> d_texMessages;
texture<unsigned int> d_texPBM;

__global__ void init_curand(curandState *state, unsigned long long seed) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < d_agentCount)
		curand_init(seed, id, 0, &state[id]);
}
__global__ void init_agents(curandState *state, glm::vec2 *locationMessages) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= d_agentCount)
		return;
	//curand_unform returns 0<x<=1.0, not much can really do about 0 exclusive
	//negate and  + 1.0, to make  0<=x<1.0
	locationMessages[id].x = (-curand_uniform(&state[id]) + 1.0f)*d_environmentWidth_float;
	locationMessages[id].y = (-curand_uniform(&state[id]) + 1.0f)*d_environmentWidth_float;
}
__device__ __forceinline__ glm::ivec2 getGridPosition(glm::vec2 worldPos)
{
	//Clamp each grid coord to 0<=x<dim
	return clamp(floor((worldPos / d_environmentWidth_float)*d_gridDim_float), glm::vec2(0), glm::vec2((float)d_gridDim - 1));
}
__device__ __forceinline__ unsigned int getHash(glm::ivec2 gridPos)
{
	//Bound gridPos to gridDimensions
	gridPos = clamp(gridPos, glm::ivec2(0), glm::ivec2(d_gridDim - 1));
	//Compute hash (effectivley an index for to a bin within the partitioning grid in this case)
	return (unsigned int)(
		(gridPos.y * d_gridDim) +					//y
		gridPos.x); 	                            //x
}
__global__ void atomicHistogram(unsigned int* bin_index, unsigned int* bin_sub_index, unsigned int *pbm_counts, glm::vec2 *messageBuffer)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;

	glm::ivec2 gridPos = getGridPosition(messageBuffer[index]);
	unsigned int hash = getHash(gridPos);
	bin_index[index] = hash;
	unsigned int bin_idx = atomicInc((unsigned int*)&pbm_counts[hash], 0xFFFFFFFF);
	bin_sub_index[index] = bin_idx;
}
__global__ void reorderLocationMessages(
	unsigned int* bin_index,
	unsigned int* bin_sub_index,
	unsigned int *pbm,
	glm::vec2 *unordered_messages,
	glm::vec2 *ordered_messages
)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;

	unsigned int i = bin_index[index];
	unsigned int sorted_index = pbm[i] + bin_sub_index[index];

	//Order messages into swap space
	ordered_messages[sorted_index] = unordered_messages[index];
}
int requiredSM(int blockSize)
{
	cudaDeviceProp dp;
	int device;
	cudaGetDevice(&device);
	memset(&dp, sizeof(cudaDeviceProp), 0);
	cudaGetDeviceProperties(&dp, device);
	//We could use dp.sharedMemPerBlock/N to improve occupancy
	return (int)min(PBM_max_count * sizeof(float2), dp.sharedMemPerBlock);//Need to limit this to the max SM
}
/**
* Kernel must be launched 1 block per bin
* This removes the necessity of __launch_bounds__(64) as all threads in block are touching the same messages
* However we end up with alot of (mostly) idle threads if one bin dense, others empty.
*/
__global__  void __launch_bounds__(64) neighbourSearch_control(const glm::vec2 *agents, glm::vec2 *out)
{
#define STRIPS
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;
	glm::vec2 pos = agents[index];
	glm::ivec2 gridPos = getGridPosition(pos);
	glm::ivec2 gridPosRelative;
	unsigned int count = 0;
	glm::vec2 average = glm::vec2(0);

	for (gridPosRelative.y = -1; gridPosRelative.y <= 1; gridPosRelative.y++)
	{//ymin to ymax
		int currentBinY = gridPos.y + gridPosRelative.y;
		if (currentBinY >= 0 && currentBinY < d_gridDim)
		{
#ifndef STRIPS
			for (gridPosRelative.x = -1; gridPosRelative.x <= 1; gridPosRelative.x++)
			{//xmin to xmax
				int currentBinX = gridPos.x + gridPosRelative.x;
				if (currentBinX >= 0 && currentBinX < d_gridDim)
				{
				//Find bin start and end
				unsigned int binHash = getHash(glm::ivec2(currentBinX, currentBinY));
				//if (binHash>d_gridDim*d_gridDim)
				//{
				//    printf("Hash: %d, gridDim: %d, pos: (%d, %d)\n", binHash, d_gridDim, tGridPos.x, tGridPos.y);
				//}
				unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
				unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#else

			int currentBinX = gridPos.x - 1;
			currentBinX = currentBinX >= 0 ? currentBinX : 0;
			unsigned int binHash = getHash(glm::ivec2(currentBinX, currentBinY));
			unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
			currentBinX = gridPos.x + 1;
			currentBinX = currentBinX < d_gridDim ? currentBinX : d_gridDim - 1;
			binHash = getHash(glm::ivec2(currentBinX, currentBinY));
			unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#endif
			//Iterate messages in range
			for (unsigned int i = binStart; i < binEnd; ++i)
			{
				if (i != index)//Ignore self
				{
					float2 message = tex1Dfetch(d_texMessages, i);
#ifndef CIRCLES
					if (distance(*(glm::vec2*)&message, pos) < d_RADIUS)
					{
						//message.z = pow(sqrt(sin(distance(message, pos))),3.1f);//Bonus compute
						average += *(glm::vec2*)&message;
						count++;
					}
#else
					glm::vec2 toLoc = (*(glm::vec2*)&message) - pos;//Difference
					float separation = length(toLoc);
					if (separation < d_RADIUS && separation > 0)
					{
						const float REPULSE_FACTOR = 0.05f;
						float k = sinf((separation / d_RADIUS)*3.141*-2)*REPULSE_FACTOR;
						toLoc /= separation;//Normalize (without recalculating seperation)
						average += k * toLoc;
						count++;
					}
#endif
				}
			}
		}
#ifndef STRIPS
		}
	}
#endif
}
average /= count>0 ? count : 1;
#ifndef CIRCLES
out[index] = average;
#else
out[index] = pos + average;
#endif
}
/**
 * Kernel must be launched 1 block per bin
 * This removes the necessity of __launch_bounds__(64) as all threads in block are touching the same messages
 * However we end up with alot of (mostly) idle threads if one bin dense, others empty.
 */
__global__ void neighbourSearch(const glm::vec2 *agents, glm::vec2 *out)
{
	extern __shared__ float2 sm_messages[];

	//My data
	glm::ivec2 myBin = glm::ivec2(blockIdx.x, blockIdx.y);
	unsigned int index = UINT_MAX;
	glm::vec2 pos;
	{
		unsigned int binHash = getHash(myBin);
		unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
		unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
		unsigned int binCount = binEnd - binStart;
		if (threadIdx.x < binCount)
		{
			index = binStart + threadIdx.x;
			pos = agents[index];
		}
	}
	//Model data
	unsigned int count = 0;
	glm::vec2 average = glm::vec2(0);
	//Iterate each bin in the Moore neighbourhood
	glm::ivec2 currentBin;
	for(int _x = -1;_x<=1;++_x)
	{
		currentBin.x = myBin.x + _x;
		if (currentBin.x >= 0 && currentBin.x < d_gridDim)
		{
			for (int _y = -1; _y <= 1; ++_y)
			{
				currentBin.y = myBin.y + _y;
				if (currentBin.y >= 0 && currentBin.y < d_gridDim)
				{
					//Now we must load all messages from currentBin into shared memory
					//WARNING: There is an unhandled edge case whereby we dont have enough shared memory, and must segment the load
					unsigned int binHash = getHash(currentBin);
					unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
					unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
					unsigned int binCount = binEnd - binStart;
					//If this bin has a message for us to load
					if(threadIdx.x<binCount)
					{
						//Load the message into shared memory
						float2 message = tex1Dfetch(d_texMessages, binStart + threadIdx.x);
						sm_messages[threadIdx.x] = message;
					}
					//Wait for all loading to be completed
					__syncthreads();
					//If we host a valid message...
					if(index != UINT_MAX)
					{
						//Iterate the loaded messages
						for (unsigned int i = 0; i<binCount;++i)
						{
							//Skip our own loaded message
							if (_x == 0 && _y == 0 && i == threadIdx.x)
								continue;
							float2 message = sm_messages[i];
#ifndef CIRCLES
							if (distance(*(glm::vec2*)&message, pos)<d_RADIUS)
							{
								//message.z = pow(sqrt(sin(distance(message, pos))),3.1f);//Bonus compute
								average += *(glm::vec2*)&message;
								count++;
							}
#else
							glm::vec2 toLoc = (*(glm::vec2*)&message) - pos;//Difference
							float separation = length(toLoc);
							if (separation < d_RADIUS && separation > 0)
							{
								const float REPULSE_FACTOR = 0.05f;
								float k = sinf((separation / d_RADIUS)*3.141*-2)*REPULSE_FACTOR;
								toLoc /= separation;//Normalize (without recalculating seperation)
								average += k * toLoc;
								count++;
							}
#endif
						}
					}
					//Wait for all processing to be completed, so that we can proceed to next bin
					__syncthreads();
				}
				//Could optimise here, by handling binHash outside the loop
				//For this would need to iterate _y on the outside, so hashes are contiguous
			}
		}
	}
	//If we have a valid message...
	if(index != UINT_MAX)
	{
		average /= count>0 ? count : 1;
#ifndef CIRCLES
		out[index] = average;
#else
		out[index] = pos + average;
#endif
	}
}


__global__ void unsortMessages(
	unsigned int* bin_index,
	unsigned int* bin_sub_index,
	unsigned int *pbm,
	glm::vec2 *ordered_messages,
	glm::vec2 *unordered_messages
)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;

	unsigned int i = bin_index[index];
	unsigned int sorted_index = pbm[i] + bin_sub_index[index];

	//Order messages into swap space
	unordered_messages[index] = ordered_messages[sorted_index];
}
/**
* This program is to act as a test rig to demonstrate the raw impact of raw message handling
*/
void run(std::ofstream &f, const unsigned int ENV_WIDTH, const unsigned int AGENT_COUNT = 1000000)
{
	void *d_CUB_temp_storage = nullptr;
	size_t d_CUB_temp_storage_bytes = 0;
	//Spatial partitioning mock
	//Fixed 2D environment of 1000x1000
	//Filled with 1,000,000 randomly distributed agents
	//const unsigned int ENV_WIDTH = 250;
	float ENV_WIDTH_float = (float)ENV_WIDTH;
	const unsigned int RNG_SEED = 12;
	const unsigned int ENV_VOLUME = ENV_WIDTH * ENV_WIDTH;
	CUDA_CALL(cudaMemcpyToSymbol(d_agentCount, &AGENT_COUNT, sizeof(unsigned int)));
	CUDA_CALL(cudaMemcpyToSymbol(d_environmentWidth_float, &ENV_WIDTH_float, sizeof(float)));
	glm::vec2 *d_agents_init = nullptr, *d_agents = nullptr, *d_out = nullptr;
	unsigned int *d_keys = nullptr, *d_vals = nullptr;
	CUDA_CALL(cudaMalloc(&d_agents_init, sizeof(glm::vec2) * AGENT_COUNT));
	CUDA_CALL(cudaMalloc(&d_agents, sizeof(glm::vec2) * AGENT_COUNT));
	CUDA_CALL(cudaMalloc(&d_out, sizeof(glm::vec2) * AGENT_COUNT));
	glm::vec2 *h_out = (glm::vec2*)malloc(sizeof(glm::vec2) * AGENT_COUNT);
	glm::vec2 *h_out_control = (glm::vec2*)malloc(sizeof(glm::vec2) * AGENT_COUNT);
	//Init agents
	{
		//Generate curand
		curandState *d_rng;
		CUDA_CALL(cudaMalloc(&d_rng, AGENT_COUNT * sizeof(curandState)));
		//Arbitrary thread block sizes (speed not too important during one off initialisation)
		unsigned int initThreads = 512;
		unsigned int initBlocks = (AGENT_COUNT / initThreads) + 1;
		init_curand << <initBlocks, initThreads >> >(d_rng, RNG_SEED);//Defined in CircleKernels.cuh
		CUDA_CALL(cudaDeviceSynchronize());
		cudaProfilerStart();//Start here because init_curand is super slow for large agent count's.
		init_agents << <initBlocks, initThreads >> >(d_rng, d_agents_init);
		//Free curand
		CUDA_CALL(cudaFree(d_rng));
		CUDA_CALL(cudaMalloc(&d_keys, sizeof(unsigned int)*AGENT_COUNT));
		CUDA_CALL(cudaMalloc(&d_vals, sizeof(unsigned int)*AGENT_COUNT));
	}
	//Decide interaction radius
	//for a range of bin widths
	const float RADIUS = 1.0f;//
	const float RADIAL_VOLUME = glm::pi<float>()*RADIUS*RADIUS;
	const unsigned int AVERAGE_NEIGHBOURS = (unsigned int)(AGENT_COUNT*RADIAL_VOLUME / ENV_VOLUME);
	printf("Agents: %d, RVol: %.2f, Average Neighbours: %d\n", AGENT_COUNT, RADIAL_VOLUME, AVERAGE_NEIGHBOURS);
	//{
	//    cudaFree(d_agents_init);
	//    cudaFree(d_agents);
	//    cudaFree(d_out);
	//    return;
	//}

	const float rSin45 = (float)(RADIUS*sin(glm::radians(45)));
	CUDA_CALL(cudaMemcpyToSymbol(d_RADIUS, &RADIUS, sizeof(float)));
	CUDA_CALL(cudaMemcpyToSymbol(d_R_SIN_45, &rSin45, sizeof(float)));
	{
		{
			//Copy init state to d_out   
			CUDA_CALL(cudaMemcpy(d_out, d_agents_init, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToDevice));
		}
		//Decide bin width (as a ratio to radius)
		const float BIN_WIDTH = RADIUS;
		float GRID_DIMS_float = ENV_WIDTH / BIN_WIDTH;
		GRID_DIMS = glm::uvec2((unsigned int)ceil(GRID_DIMS_float));
		CUDA_CALL(cudaMemcpyToSymbol(d_binWidth, &BIN_WIDTH, sizeof(float)));
		CUDA_CALL(cudaMemcpyToSymbol(d_gridDim, &GRID_DIMS.x, sizeof(unsigned int)));
		CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &GRID_DIMS_float, sizeof(float)));
		const unsigned int BIN_COUNT = glm::compMul(GRID_DIMS);
		cudaEvent_t start_PBM, end_PBM, start_kernel, end_kernel;
		cudaEventCreate(&start_PBM);
		cudaEventCreate(&end_PBM);
		cudaEventCreate(&start_kernel);
		cudaEventCreate(&end_kernel);
		//BuildPBM
		unsigned int *d_PBM_counts = nullptr;
		unsigned int *d_PBM = nullptr;
		CUDA_CALL(cudaMalloc(&d_PBM_counts, (BIN_COUNT + 1) * sizeof(unsigned int)));
		CUDA_CALL(cudaMalloc(&d_PBM, (BIN_COUNT + 1) * sizeof(unsigned int)));
		//Prep for threadblocks
		CUDA_CALL(cudaMalloc(&d_PBM_max_count, sizeof(unsigned int)));
		CUDA_CALL(cudaMemset(d_PBM_max_count, 0, sizeof(unsigned int)));
		{//Resize cub temp if required
			size_t bytesCheck, bytesCheck2;
			cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, d_PBM, d_PBM_counts, BIN_COUNT + 1);
			cub::DeviceReduce::Max(nullptr, bytesCheck2, d_PBM_counts, d_PBM_max_count, BIN_COUNT);
			bytesCheck = glm::max(bytesCheck, bytesCheck2);
			if (bytesCheck > d_CUB_temp_storage_bytes)
			{
				if (d_CUB_temp_storage)
				{
					CUDA_CALL(cudaFree(d_CUB_temp_storage));
				}
				d_CUB_temp_storage_bytes = bytesCheck;
				CUDA_CALL(cudaMalloc(&d_CUB_temp_storage, d_CUB_temp_storage_bytes));
			}
		}

		float pbmMillis_control = 0, kernelMillis_control = 0;
		float pbmMillis = 0, kernelMillis = 0;
		for (unsigned int _j = 1; _j < UINT_MAX; --_j)
		{
			//1 = control
			//0 = threadblock
			bool isControl = _j != 0;

			//For 200 iterations (to produce an average)
			const unsigned int ITERATIONS = 1;
			for (unsigned int i = 0; i < ITERATIONS; ++i)
			{
				//Reset each run of average model
#ifndef CIRCLES
				CUDA_CALL(cudaMemcpy(d_out, d_agents_init, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToDevice));
#endif	
				cudaEventRecord(start_PBM);
				{//Build atomic histogram
					CUDA_CALL(cudaMemset(d_PBM_counts, 0x00000000, (BIN_COUNT + 1) * sizeof(unsigned int)));
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, atomicHistogram, 32, 0));//Randomly 32
																												 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					atomicHistogram << <gridSize, blockSize >> > (d_keys, d_vals, d_PBM_counts, d_out);
					CUDA_CALL(cudaDeviceSynchronize());
				}
				{//Scan (sum), to finalise PBM
					cub::DeviceScan::ExclusiveSum(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_PBM_counts, d_PBM, BIN_COUNT + 1);
				}
				{//Reorder messages
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																														 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					//Copy messages from d_messages to d_messages_swap, in hash order
					reorderLocationMessages << <gridSize, blockSize >> > (d_keys, d_vals, d_PBM, d_out, d_agents);
					CUDA_CHECK();
				}
				if (!isControl)
				{//Calc max bin size (for threadblocks)
					cub::DeviceReduce::Max(d_CUB_temp_storage, d_CUB_temp_storage_bytes, d_PBM_counts, d_PBM_max_count, BIN_COUNT);
					CUDA_CALL(cudaGetLastError());
					CUDA_CALL(cudaMemcpy(&PBM_max_count, d_PBM_max_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
					//Calc moore size (bin size^dims?)
					//PBM_max_Moore_count = (unsigned int)pow(PBM_max_count, 2);//2==2D//Unused, requires 9x shared mem in 2D, 27x in 3D
				}
				{//Fill PBM and Message Texture Buffers																			  
					CUDA_CALL(cudaDeviceSynchronize());//Wait for return
					CUDA_CALL(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec2) * AGENT_COUNT));
					CUDA_CALL(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (BIN_COUNT + 1)));
				}
				cudaEventRecord(end_PBM);
				cudaEventRecord(start_kernel);
				if (isControl)
				{
					//Each message samples radial neighbours (static model)
					int blockSize;   // The launch configurator returned block size 
					CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
					 // Round up according to array size
					int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
					//Copy messages from d_agents to d_out, in hash order
					neighbourSearch_control << <gridSize, blockSize >> > (d_agents, d_out);
					CUDA_CHECK();
				}
				else
				{
					//Each message samples radial neighbours (static model)
					int blockSize = PBM_max_count;   //blockSize == largest bin size
					dim3 gridSize;
					gridSize.x = GRID_DIMS.x;
					gridSize.y = GRID_DIMS.y;
					gridSize.z = 1;// GRID_DIMS.z;
					//Copy messages from d_agents to d_out, in hash order
					neighbourSearch << <gridSize, blockSize, requiredSM(blockSize) >> > (d_agents, d_out);
					CUDA_CHECK();
				}
				CUDA_CALL(cudaDeviceSynchronize());
				cudaEventRecord(end_kernel);
				cudaEventSynchronize(end_kernel);

				float _pbmMillis = 0, _kernelMillis = 0;
				cudaEventElapsedTime(&_pbmMillis, start_PBM, end_PBM);
				cudaEventElapsedTime(&_kernelMillis, start_kernel, end_kernel);
				if (isControl)
				{
					pbmMillis_control += _pbmMillis;
					kernelMillis_control += _kernelMillis;
				}
				else
				{
					pbmMillis += _pbmMillis;
					kernelMillis += _kernelMillis;
				}

			}//for(ITERATIONS)
			pbmMillis_control /= ITERATIONS;
			kernelMillis_control /= ITERATIONS;
			pbmMillis /= ITERATIONS;
			kernelMillis /= ITERATIONS;

			{//Unorder messages
				int blockSize;   // The launch configurator returned block size 
				CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																													 // Round up according to array size
				int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
				//Copy messages from d_out to d_agents, in hash order
				unsortMessages << <gridSize, blockSize >> > (d_keys, d_vals, d_PBM, d_out, d_agents);
				CUDA_CHECK();
				//Swap d_out and d_agents
				{
					glm::vec2 *t = d_out;
					d_out = d_agents;
					d_agents = t;
				}
				//Wait for return
				CUDA_CALL(cudaDeviceSynchronize());
				//Copy back to relative host array (for validation)
				CUDA_CALL(cudaMemcpy(isControl ? h_out_control : h_out, d_out, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToHost));
				CUDA_CALL(cudaDeviceSynchronize());
			}
		}//for(MODE)
		CUDA_CALL(cudaUnbindTexture(d_texPBM));
		CUDA_CALL(cudaUnbindTexture(d_texMessages));
		CUDA_CALL(cudaFree(d_PBM_counts));
		CUDA_CALL(cudaFree(d_PBM));
		//log();
		printf("Control:     PBM: %.2fms, Kernel: %.2fms\n", pbmMillis_control, kernelMillis_control);
		printf("ThreadBlock: PBM: %.2fms, Kernel: %.2fms\n", pbmMillis, kernelMillis);
		unsigned int fails = 0;
		cudaProfilerStop();
#ifndef CIRCLES
		{//Validation
			//Validate results for average model
			//thrust::sort(thrust::cuda::par, d_out, d_out + AGENT_COUNT, vec2Compare());
			//CUDA_CALL(cudaMemcpy(isControl ? h_out_control : h_out, d_out, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToHost));
			for (unsigned int i = 0; i < AGENT_COUNT; ++i)
			{
				assert(!(isnan(h_out[i].x) || isnan(h_out[i].y)));
				if (isnan(h_out[i].x) || isnan(h_out[i].y))
					printf("err nan\n");
				auto ret = glm::epsilonEqual(h_out[i], h_out_control[i], EPSILON);
				if (!(ret.x&&ret.y))
				{
					if (fails == 0)
						printf("(%.5f, %.5f) vs (%.5f, %.5f)\n", h_out_control[i].x, h_out_control[i].y, h_out[i].x, h_out[i].y);
					fails++;
				}
			}
			if (fails > 0)
				printf("%d/%d (%.1f%%) Failed.\n", fails, AGENT_COUNT, 100 * (fails / (float)AGENT_COUNT));
			else
				printf("Validation passed %d/%d\n", AGENT_COUNT, AGENT_COUNT);
		}
#else
		printf("Validation not ran for Circles mode.\n")
#endif
		log(f, AVERAGE_NEIGHBOURS, AGENT_COUNT, ENV_WIDTH, pbmMillis_control, kernelMillis_control, pbmMillis, kernelMillis, fails);
	}

	CUDA_CALL(cudaUnbindTexture(d_texMessages));
	CUDA_CALL(cudaFree(d_vals));
	CUDA_CALL(cudaFree(d_keys));
	CUDA_CALL(cudaFree(d_agents));
	CUDA_CALL(cudaFree(d_agents_init));
	CUDA_CALL(cudaFree(d_out));
	free(h_out);
	free(h_out_control);
	cudaDeviceReset();
}
void runAgents(std::ofstream &f, const unsigned int AGENT_COUNT, const float DENSITY)
{
	//density refers to approximate number of neighbours
	run(f, (unsigned int)sqrt(AGENT_COUNT / (DENSITY*2.86 / 9)), AGENT_COUNT);
}
int main()
{
	{
		std::ofstream f;
		createLog(f);
		assert(f.is_open());
		//for (unsigned int i = 20000; i <= 3000000; i += 20000)
		for (unsigned int i = 100000; i <= 100000; i += 20000)
		{
			//Run i agents in a density with roughly 60 radial neighbours, and log
			//Within this, it is tested over a range of proportional bin widths
			runAgents(f, i, 650);
			//break;
		}
	}
	printf("fin\n");
	//getchar();
	return 0;
}


#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>
#include <cub/cub.cuh>
#include <glm/gtc/epsilon.hpp>
#define EPSILON 0.005f
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
	const float &rad,
	const float &binWidth,
	const unsigned int &estRadialNeighbours,
	const unsigned int &agentCount,
	const unsigned int &envWidth,
	const float &PBM,
	const float &kernel,
	const unsigned int &fails
);
__device__ __constant__ unsigned int d_agentCount;
__device__ __constant__ float d_environmentWidth_float;
__device__ __constant__ unsigned int d_gridDim;
__device__ __constant__ float d_gridDim_float;
__device__ __constant__ float d_RADIUS;
__device__ __constant__ float d_R_SIN_45;
__device__ __constant__ float d_binWidth;

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
__global__ void __launch_bounds__(64) neighbourSearch(const glm::vec2 *agents, glm::vec2 *out)
{
#define STRIPS
#define CIRCLES
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//Kill excess threads
	if (index >= d_agentCount) return;
	glm::vec2 pos = agents[index];
	glm::ivec2 tGridPos;
	glm::ivec2 topLeft = getGridPosition(pos - d_RADIUS);
	glm::ivec2 bottomRight = getGridPosition(pos + d_RADIUS);
	unsigned int count = 0;
	glm::vec2 average = glm::vec2(0);
	for (tGridPos.y = topLeft.y; tGridPos.y <= bottomRight.y; tGridPos.y++)
	{//xmin to xmax
#ifndef STRIPS
		for (tGridPos.x = topLeft.x; tGridPos.x <= bottomRight.x; tGridPos.x++)
		{//ymin to ymax
		 //Find bin start and end
			unsigned int binHash = getHash(tGridPos);
			//if (binHash>d_gridDim*d_gridDim)
			//{
			//    printf("Hash: %d, gridDim: %d, pos: (%d, %d)\n", binHash, d_gridDim, tGridPos.x, tGridPos.y);
			//}
			unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
			unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#else
		unsigned int binHash = getHash(glm::ivec2(topLeft.x, tGridPos.y));
		unsigned int binStart = tex1Dfetch(d_texPBM, binHash);
		binHash = getHash(glm::ivec2(bottomRight.x, tGridPos.y));
		unsigned int binEnd = tex1Dfetch(d_texPBM, binHash + 1);
#endif
		//Iterate messages in range
		for (unsigned int i = binStart; i < binEnd; ++i)
		{
			if (i != index)//Ignore self
			{
				float2 message = tex1Dfetch(d_texMessages, i);
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
#ifndef STRIPS
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
	glm::vec2 *h_out0 = (glm::vec2*)malloc(sizeof(glm::vec2) * AGENT_COUNT);
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
	//for (unsigned int k = 7; k < 8; ++k)
	//for (unsigned int k = 0; k < 16; ++k)
	for (unsigned int k = 0; k < 2; ++k)
	{
		float binRatio = 1.5f - (k*0.1f);
		if (k == 0)
			binRatio = 1.0f;
		if (k == 1)
			binRatio = 0.5f;
		if (k == 14)//Special case, final iteration test 2/3
			binRatio = 2.0f / 3;
		if (k == 15)//Special case, final iteration test 2/3
			binRatio = 1.0f / 3;
		{
			//Copy init state to d_out   
			CUDA_CALL(cudaMemcpy(d_out, d_agents_init, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToDevice));
		}
		//Decide bin width (as a ratio to radius)
		const float BIN_WIDTH = RADIUS*binRatio;
		float GRID_DIMS_float = ENV_WIDTH / BIN_WIDTH;
		const glm::uvec2 GRID_DIMS = glm::uvec2((unsigned int)ceil(GRID_DIMS_float));
		CUDA_CALL(cudaMemcpyToSymbol(d_binWidth, &BIN_WIDTH, sizeof(float)));
		CUDA_CALL(cudaMemcpyToSymbol(d_gridDim, &GRID_DIMS.x, sizeof(unsigned int)));
		CUDA_CALL(cudaMemcpyToSymbol(d_gridDim_float, &GRID_DIMS_float, sizeof(float)));
		const unsigned int BIN_COUNT = glm::compMul(GRID_DIMS);
		if (BIN_COUNT > 20000000)continue;//Smaller ratio's lead to this value becoming invalid large (errors at tex bind of pbm)
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
		{//Resize cub temp if required
			size_t bytesCheck;
			cub::DeviceScan::ExclusiveSum(nullptr, bytesCheck, d_PBM, d_PBM_counts, BIN_COUNT + 1);
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

		//For 200 iterations (to produce an average)
		float pbmMillis = 0, kernelMillis = 0;
		const unsigned int ITERATIONS = 200;
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
				atomicHistogram << <gridSize, blockSize >> >(d_keys, d_vals, d_PBM_counts, d_out);
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
				reorderLocationMessages << <gridSize, blockSize >> >(d_keys, d_vals, d_PBM, d_out, d_agents);
				CUDA_CHECK();
				//Wait for return
				CUDA_CALL(cudaDeviceSynchronize());
			}
			{//Fill PBM and Message Texture Buffers
				CUDA_CALL(cudaBindTexture(nullptr, d_texMessages, d_agents, sizeof(glm::vec2) * AGENT_COUNT));
				CUDA_CALL(cudaBindTexture(nullptr, d_texPBM, d_PBM, sizeof(unsigned int) * (BIN_COUNT + 1)));
			}
			cudaEventRecord(end_PBM);

			cudaEventRecord(start_kernel);
			{
				//Each message samples radial neighbours (static model)
				int blockSize;   // The launch configurator returned block size 
				CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																													 // Round up according to array size
				int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
				//Copy messages from d_messages to d_messages_swap, in hash order
				neighbourSearch << <gridSize, blockSize >> >(d_agents, d_out);
				CUDA_CHECK();
			}
			CUDA_CALL(cudaDeviceSynchronize());
			cudaEventRecord(end_kernel);
			cudaEventSynchronize(end_kernel);
			float _pbmMillis = 0, _kernelMillis = 0;
			cudaEventElapsedTime(&_pbmMillis, start_PBM, end_PBM);
			cudaEventElapsedTime(&_kernelMillis, start_kernel, end_kernel);
			pbmMillis += _pbmMillis;
			kernelMillis += _kernelMillis;
		}
		pbmMillis /= ITERATIONS;
		kernelMillis /= ITERATIONS;
		{//Unorder messages
			int blockSize;   // The launch configurator returned block size 
			CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, reorderLocationMessages, 32, 0));//Randomly 32
																												 // Round up according to array size
			int gridSize = (AGENT_COUNT + blockSize - 1) / blockSize;
			//Copy messages from d_messages to d_messages_swap, in hash order
			unsortMessages << <gridSize, blockSize >> >(d_keys, d_vals, d_PBM, d_out, d_agents);
			CUDA_CHECK();
			glm::vec2 *t = d_out;
			d_out = d_agents;
			d_agents = t;
			//Wait for return
			CUDA_CALL(cudaDeviceSynchronize());
		}
		CUDA_CALL(cudaUnbindTexture(d_texPBM));
		CUDA_CALL(cudaUnbindTexture(d_texMessages));
		CUDA_CALL(cudaFree(d_PBM_counts));
		CUDA_CALL(cudaFree(d_PBM));
		//log();
		printf("BW: x%.2f, PBM: %.2fms, Kernel: %.2fms\n", binRatio, pbmMillis, kernelMillis);
		unsigned int fails = 0;
#ifndef CIRCLES
		//Validate results for average model
		//thrust::sort(thrust::cuda::par, d_out, d_out + AGENT_COUNT, vec2Compare());
		CUDA_CALL(cudaMemcpy(k == 0 ? h_out : h_out0, d_out, sizeof(glm::vec2)*AGENT_COUNT, cudaMemcpyDeviceToHost));
		if (k != 0)
		{
			for (unsigned int i = 0; i < AGENT_COUNT; ++i)
			{
				assert(!(isnan(h_out[i].x) || isnan(h_out[i].y)));
				if (isnan(h_out[i].x) || isnan(h_out[i].y))
					printf("err nan\n");
				auto ret = glm::epsilonEqual(h_out[i], h_out0[i], EPSILON);
				if (!(ret.x&&ret.y))
				{
					if (fails == 0)
						printf("(%.5f, %.5f) vs (%.5f, %.5f)\n", h_out[i].x, h_out[i].y, h_out0[i].x, h_out0[i].y);
					fails++;
				}
			}
			if (fails > 0)
				printf("%d/%d (%.1f%%) Failed.\n", fails, AGENT_COUNT, 100 * (fails / (float)AGENT_COUNT));
		}
#endif
		log(f, RADIUS, BIN_WIDTH, AVERAGE_NEIGHBOURS, AGENT_COUNT, ENV_WIDTH, pbmMillis, kernelMillis, fails);
	}

	CUDA_CALL(cudaUnbindTexture(d_texMessages));
	CUDA_CALL(cudaFree(d_vals));
	CUDA_CALL(cudaFree(d_keys));
	CUDA_CALL(cudaFree(d_agents));
	CUDA_CALL(cudaFree(d_agents_init));
	CUDA_CALL(cudaFree(d_out));
	free(h_out);
	free(h_out0);
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
		for (unsigned int i = 20000; i <= 3000000; i += 20000)
		{
			//Run i agents in a density with roughly 60 radial neighbours, and log
			//Within this, it is tested over a range of proportional bin widths
			runAgents(f, i, 60);
		}
	}
	printf("fin\n");
	getchar();
	return 0;
}


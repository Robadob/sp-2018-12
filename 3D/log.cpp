#define _CRT_SECURE_NO_WARNINGS
#include <ctime>
#include <string>
#include <fstream>
void createLog(std::ofstream &f)
{
    std::string path = std::string("[3D]") + std::to_string(time(nullptr)) + ".csv";
    f.open(path);

    //Write header
    f << "Est Radial Neighbours,";
    f << "Agent Count,";
    f << "Env Width,";
	f << "PBM_control (ms),";
	f << "Kernel_control (ms),";
    f << "PBM (ms),";
    f << "Kernel (ms),";
    f << "Failures,";
    f << "\n";
}
void log(std::ofstream &f,
    const unsigned int &estRadialNeighbours,
    const unsigned int &agentCount,
    const unsigned int &envWidth,
	const float &PBM_control,
	const float &kernel_control,
    const float &PBM,
    const float &kernel,
    const unsigned int &fails
    )
{
    char buffer[1024];

    sprintf(&buffer[0], "%u,", estRadialNeighbours);
    f << buffer;
    sprintf(&buffer[0], "%u,", agentCount);
    f << buffer;
    sprintf(&buffer[0], "%u,", envWidth);
    f << buffer;
	sprintf(&buffer[0], "%f,", PBM_control);
	f << buffer;
	sprintf(&buffer[0], "%f,", kernel_control);
	f << buffer;
    sprintf(&buffer[0], "%f,", PBM);
    f << buffer;
    sprintf(&buffer[0], "%f,", kernel);
    f << buffer;
    sprintf(&buffer[0], "%u,", fails);
    f << buffer;
    f << "\n";
}

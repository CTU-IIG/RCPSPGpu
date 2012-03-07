#ifndef HLIDAC_PES_CUDA_FUNCTIONS_CUH
#define HLIDAC_PES_CUDA_FUNCTIONS_CUH

#include <cuda.h>
#include <stdint.h>

enum TextureName	{
	ACTIVITIES_RESOURCES = 0,
	PREDECESSORS = 1,
	PREDECESSORS_INDICES = 2
};

struct MoveIndices	{
	int16_t i;
	int16_t j;
};

struct SolutionInfo {
	uint32_t solutionCost;
	uint32_t readCounter;
};

struct MoveInfo	{
	int16_t i;
	int16_t j;
	uint32_t cost;
};

struct CudaData {
	int16_t numberOfActivities;
	int8_t numberOfResources;

	uint16_t swapRange;
	uint16_t sumOfCapacities;
	uint8_t maximalCapacityOfResource;
	uint32_t numberOfIterationsPerBlock;
	uint32_t maximalIterationsSinceBest;

	uint8_t *activitiesDuration;

	uint8_t *successorsMatrix;
	uint32_t successorsMatrixSize;
	bool copySuccessorsMatrixToSharedMemory;

	uint16_t *resourceIndices;

	uint32_t maxTabuListSize;
	MoveIndices *tabuLists;
	uint8_t *tabuCaches;

	uint32_t *hashMap;
	bool useTabuHash;

	uint32_t solutionsSetSize;
	uint16_t *solutionsSet;
	MoveIndices *solutionSetTabuLists;
	SolutionInfo *solutionsSetInfo;
	uint32_t *setStateOfCommunication;

	uint16_t *globalBestSolution;
	uint32_t *globalBestSolutionCost;
	MoveIndices *globalBestSolutionTabuList;
	uint32_t *globalStateOfCommunication;

	uint16_t *blocksBestSolution;
	
	MoveIndices *mergeHelpArray;
	MoveIndices *swapFreeMergeArray;

	uint32_t maximalValueOfReadCounter;
	uint32_t numberOfDiversificationSwaps;
};

int bindTexture(void *texData, int32_t arrayLength, int option);
int unbindTexture(int option);

void runCudaSolveRCPSP(int numberOfBlock, int numberOfThreadsPerBlock, int computeCapability, int dynSharedMemSize, const CudaData& cudaData);

#endif


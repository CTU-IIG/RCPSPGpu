#ifndef HLIDAC_PES_CUDA_FUNCTIONS_CUH
#define HLIDAC_PES_CUDA_FUNCTIONS_CUH

#include <cuda.h>
#include <stdint.h>

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
	int16_t numberOfActivities;	// OK
	int16_t numberOfResources;	// OK

	uint32_t swapRange;	// OK
	uint16_t sumOfCapacities;	// OK
	uint8_t maximalCapacityOfResource;	// OK
	uint32_t numberOfIterationsPerBlock;	// OK
	uint32_t maximalIterationsSinceBest;	// OK

	int32_t maxTabuListSize;	// OK
	MoveIndices *tabuLists;		// OK
	uint8_t *tabuCaches;	// OK

	int32_t solutionsSetSize;	// OK
	uint16_t *solutionsSet;		// OK
	MoveIndices *solutionSetTabuLists;	// OK
	SolutionInfo *solutionsSetInfo;	// OK
	uint32_t *setStateOfCommunication;	// OK

	uint16_t *globalBestSolution;	// OK
	uint32_t *globalBestSolutionCost;	// OK
	MoveIndices *globalBestSolutionTabuList;	// OK
	uint32_t *globalStateOfCommunication;	// OK

	uint16_t *blocksBestSolution;	//	OK
	
	uint8_t *activitiesDuration;	// OK

	uint8_t *successorsMatrix;		// OK
	uint32_t successorsMatrixSize;	// OK
	bool copySuccessorsMatrixToSharedMemory;	// OK
	
	MoveIndices *mergeHelpArray;	// OK
	MoveIndices *swapFreeMergeArray;	// OK

	uint16_t *resourceIndices;	// OK

	uint32_t *hashMap; // OK

	uint32_t maximalValueOfReadCounter;
	uint32_t numberOfDiversificationSwaps;
};

int bindTexture(void *texData, int32_t arrayLength, int option);
int unbindTexture(int option);

void runCudaSolveRCPSP(int numberOfBlock, int numberOfThreadsPerBlock, int computeCapability, int dynSharedMemSize, const CudaData& cudaData);

#endif


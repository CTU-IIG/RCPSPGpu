#ifndef HLIDAC_PES_CUDA_FUNCTIONS_CUH
#define HLIDAC_PES_CUDA_FUNCTIONS_CUH

/*!
 * Define basic data structures, entry point for a global function, texture helper functions.
 * \file CudaFunctions.cuh
 * \author Libor Bukata
 * \brief Basic Cuda data structures are defined at this file. 
 */

#include <stdint.h>

/* CUDA DATA STRUCTURES */ 

//! Define basic names of textures.
enum TextureName	{
	ACTIVITIES_RESOURCES = 0,
	PREDECESSORS = 1,
	PREDECESSORS_INDICES = 2,
};

//! It defines the names of the structures in the device constant memory.
enum ConstantMemoryNames	{
	THE_LONGEST_PATHS = 0
};

/*!
 * \struct MoveIndices
 * \brief Move indices are stored at this structure.
 */
struct MoveIndices	{
	//! Index i at order of activities.
	int16_t i;
	//! Index j at order of activities.
	int16_t j;
};

/*!
 * \struct SolutionInfo
 * \brief Info structure of the one solution from the set.
 */
struct SolutionInfo {
	//! Cost of the solution.
	uint32_t solutionCost;
	//! How many times was the solution read without improving of the cost.
	uint32_t readCounter;
	//! The number of iterations performed at this solution.
	uint32_t iterationCounter;
};

/*!
 * \struct MoveInfo
 * \brief Structure that is used for every block thread to remember best found solution.
 */
struct MoveInfo	{
	//! Index i at order of activities.
	int16_t i;
	//! Index j at order of activities.
	int16_t j;
	//! Best found cost.
	uint32_t cost;
};

/*!
 * \struct CudaData
 * \brief Basic constant variables, pointers to required data and configure parameters.
 */
struct CudaData {
	//! Number of instance activities.
	int16_t numberOfActivities;
	//! Total number of resources.
	int8_t numberOfResources;

	//! Maximal distance between two swapped activities.
	uint16_t swapRange;
	//! Sum of the capacities of resources.
	uint16_t sumOfCapacities;
	//! Maximal capacity of resource.
	uint8_t maximalCapacityOfResource;
	//! Number of iterations per one block.
	uint32_t numberOfIterationsPerBlock;

	//! Duration of activities.
	uint8_t *durationOfActivities;

	//! Successors bit matrix. 
	uint8_t *successorsMatrix;
	//! Successors matrix size. (bytes)
	uint32_t successorsMatrixSize;
	//! Copy successors matrix to device shared memory?
	bool copySuccessorsMatrixToSharedMemory;

	//! Indices of resources.
	uint16_t *resourceIndices;

	//! Upper bound of tabu list size.
	uint32_t maxTabuListSize;
	//! Tabu list for every block.
	MoveIndices *tabuLists;
	//! Tabu cache for every block.
	uint8_t *tabuCaches;

	//! Number of solutions at the solution set.
	uint32_t totalSolutions;
	//! Solutions (= orders) at set.
	uint16_t *ordersOfSolutions;
	//! Info about each solution at set.
	SolutionInfo *infoAboutSolutions;
	//! Tabu list for each solution at set.
	MoveIndices *tabuListsOfSetOfSolutions;
	//! Lock variable - access to set solutions.
	uint32_t *lockSetOfSolutions;

	//! The cost of the best found solution.
	uint32_t *bestSolutionCost;
	//! An index to the best solution.
	uint32_t *indexToTheBestSolution;

	//! Every block save improving solution to proper order.
	uint16_t *blocksBestSolution;
	
	//! Help arrays for reorder of moves.
	MoveIndices *mergeHelpArray;
	//! Arrays for reorder of precedence penalty free moves.
	MoveIndices *swapMergeArray;

	//! How many times can be solution read without improving than diversification will be called.
	uint32_t maximalValueOfReadCounter;
	//! Number of diversification swaps.
	uint32_t numberOfDiversificationSwaps;

	//! Number of evaluated schedules.
	uint64_t *evaluatedSchedules;

	//! Switch that select evaluation algorithm for the resources (time resolution or capacity resolution).
	bool capacityResolutionAlgorithm;

	//! The length of the critical path without resources restrictions.
	uint32_t criticalPathLength;
};

/* TEXTURE HELPER FUNCTIONS */

/*!
 * \param texData Array that should be bound with a texture.
 * \param arrayLength Length of the texData array.
 * \param option Texture name, see enum TextureName.
 * \return Cuda error code or cudaSuccess.
 * \brief Function bind a array with a specified texture.
 */
int bindTexture(void *texData, int32_t arrayLength, int option);
/*!
 * \param option Texture name, see enum TextureName.
 * \return Cuda error code or cudaSuccess.
 * \brief Unbind a specified texture.
 */
int unbindTexture(int option);

/* CONSTANT MEMORY HELPER FUNCTION */

/*!
 * \param source The array of values which will be copied to the constant device memory.
 * \param arrayLength The length of the source array.
 * \param option The name of the structure to bind with.
 * \return Cuda error code or cuda success.
 * \brief The function copies input array to the GPU constant memory.
 */
int memcpyToSymbol(void *source, int32_t arrayLength, int option);

/* RUN GLOBAL FUNCTION */

/*!
 * \param numberOfBlock Total number of block which will be launched.
 * \param numberOfThreadsPerBlock Number of threads per one block.
 * \param computeCapability Compute capability of graphics card. (value 210 correspond to capability 2.1)
 * \param dynSharedMemSize Amount of required shared memory.
 * \param cudaData Pointers to Cuda data, constants, configure informations, etc.
 * \brief Global kernel is called through this function. GPU solution of RCPSP will be stored at device memory.
 */
void runCudaSolveRCPSP(int numberOfBlock, int numberOfThreadsPerBlock, int computeCapability, int dynSharedMemSize, const CudaData& cudaData);

#endif


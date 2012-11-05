#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iterator>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <stdexcept>

#ifdef __GNUC__
#include <sys/time.h>
#elif defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#ifndef UINT32_MAX
#define UINT32_MAX 0xffffffff
#endif

#include "ConfigureRCPSP.h"
#include "CudaConstants.h"
#include "ScheduleSolver.cuh"
#include "SourcesLoad.h"

#if DEBUG_TABU_HASH == 1
#include <map>
#endif

using namespace std;

ScheduleSolver::ScheduleSolver(const InputReader& rcpspData, bool verbose) : solutionComputed(false), totalRunTime(-1)	{
	// Copy pointers to data of instance.
	numberOfResources = rcpspData.getNumberOfResources();
	capacityOfResources = rcpspData.getCapacityOfResources();
	numberOfActivities = rcpspData.getNumberOfActivities();
	activitiesDuration = rcpspData.getActivitiesDuration();
	numberOfSuccessors = rcpspData.getActivitiesNumberOfSuccessors();
	activitiesSuccessors = rcpspData.getActivitiesSuccessors();
	activitiesResources = rcpspData.getActivitiesResources();

	// It computes the estimate of the longest duration of the project.
	upperBoundMakespan = 0;
	for (uint16_t id = 0; id < numberOfActivities; ++id)
		upperBoundMakespan += activitiesDuration[id];
	
	// Create required structures and copy data to GPU.
	uint16_t *activitiesOrder = new uint16_t[numberOfActivities];
	createInitialSolution(activitiesOrder);
	if (prepareCudaMemory(activitiesOrder, verbose) == true)	{
		for (uint16_t i = 0; i < numberOfActivities; ++i)	{
			delete[] activitiesPredecessors[i];
		}
		delete[] activitiesPredecessors;
		delete[] numberOfPredecessors;
		throw runtime_error("ScheduleSolver::ScheduleSolver: Cuda error detected!");
	}	else	{
		bestScheduleOrder = new uint16_t[numberOfActivities];
		if (verbose == true)
			cout<<"All required resources allocated..."<<endl<<endl;
	}
}

void ScheduleSolver::createInitialSolution(uint16_t *activitiesOrder)	{

	/* PRECOMPUTE ACTIVITIES PREDECESSORS */

	activitiesPredecessors = new uint16_t*[numberOfActivities];
	numberOfPredecessors = new uint16_t[numberOfActivities];
	memset(numberOfPredecessors, 0, sizeof(uint16_t)*numberOfActivities);

	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		for (uint16_t successorIdx = 0; successorIdx < numberOfSuccessors[activityId]; ++successorIdx)	{
			uint16_t successorId = activitiesSuccessors[activityId][successorIdx];
			++numberOfPredecessors[successorId];
		}
	}

	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		activitiesPredecessors[activityId] = new uint16_t[numberOfPredecessors[activityId]];
	}

	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		for (uint16_t successorIdx = 0; successorIdx < numberOfSuccessors[activityId]; ++successorIdx)	{
			uint16_t successorId = activitiesSuccessors[activityId][successorIdx];
			*(activitiesPredecessors[successorId]) = activityId;	
			++activitiesPredecessors[successorId];
		}
	}

	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		activitiesPredecessors[activityId] -= numberOfPredecessors[activityId];
	}


	/* CREATE INIT ORDER OF ACTIVITIES */

	uint16_t deep = 0;
	uint16_t *levels = new uint16_t[numberOfActivities];
	memset(levels, 0, sizeof(uint16_t)*numberOfActivities);

		
	// Add first task with id 0. (currentLevel contain ID's)
	uint8_t *currentLevel = new uint8_t[numberOfActivities];
	uint8_t *newCurrentLevel = new uint8_t[numberOfActivities];
	memset(currentLevel, 0, sizeof(uint8_t)*numberOfActivities);

	currentLevel[0] = 1;
	bool anyActivity = true;

	while (anyActivity == true)	{
		anyActivity = false;
		memset(newCurrentLevel, 0, sizeof(uint8_t)*numberOfActivities);
		for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
			if (currentLevel[activityId] == 1)	{
				for (uint16_t nextLevelIdx = 0; nextLevelIdx < numberOfSuccessors[activityId]; ++nextLevelIdx)	{
					newCurrentLevel[activitiesSuccessors[activityId][nextLevelIdx]] = 1;
					anyActivity = true;
				}
				levels[activityId] = deep;
			}
		}

		swap(currentLevel, newCurrentLevel);

		++deep;
	}

	// Current schedule index.
	uint16_t schedIdx = 0;
	for (uint16_t curDeep = 0; curDeep < deep; ++curDeep)	{
		for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
			if (levels[activityId] == curDeep)	{
				activitiesOrder[schedIdx++] = activityId;
			}
		}
	}

	delete[] levels;
	delete[] currentLevel;
	delete[] newCurrentLevel;
}

bool ScheduleSolver::prepareCudaMemory(uint16_t *activitiesOrder, bool verbose)	{

	/* PREPARE DATA PHASE */

	/* CONVERT PREDECESSOR AND SUCCESSOR ARRAY TO 1D */
	uint32_t numOfElPred = 0, numOfElSuc = 0;
	uint16_t *predIdxs = new uint16_t[numberOfActivities+1], *sucIdxs = new uint16_t[numberOfActivities+1];

	predIdxs[0] = sucIdxs[0] = 0;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		numOfElSuc += numberOfSuccessors[i];
		numOfElPred += numberOfPredecessors[i];
		sucIdxs[i+1] = numOfElSuc;
		predIdxs[i+1] = numOfElPred;
	}

	uint16_t *linSucArray = new uint16_t[numOfElSuc], *linPredArray = new uint16_t[numOfElPred];
	uint16_t *sucWr = linSucArray, *predWr = linPredArray;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		for (uint16_t j = 0; j < numberOfPredecessors[i]; ++j)
			*(predWr++) = activitiesPredecessors[i][j];
		for (uint16_t j = 0; j < numberOfSuccessors[i]; ++j)
			*(sucWr++) = activitiesSuccessors[i][j];
	}

	/* CONVERT ACTIVITIES RESOURCE REQUIREMENTS TO 1D ARRAY */
	uint32_t resourceReqSize = numberOfActivities*numberOfResources;
	uint8_t *reqResLin = new uint8_t[resourceReqSize], *resWr = reqResLin;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		for (uint8_t r = 0; r < numberOfResources; ++r)	{
			*(resWr++) = activitiesResources[i][r];
		}
	}

	/* CONVERT CAPACITIES OF RESOURCES TO 1D ARRAY */
	uint16_t *resIdxs = new uint16_t[numberOfResources+1];
	resIdxs[0] = 0;
	for (uint8_t r = 0; r < numberOfResources; ++r)	{
		resIdxs[r+1] =  resIdxs[r]+capacityOfResources[r];
	}

	/* CREATE SUCCESSORS MATRIX + COMPUTE CRITICAL PATH MAKESPAN */
	uint32_t successorsMatrixSize = numberOfActivities*numberOfActivities/8;
	if ((numberOfActivities*numberOfActivities) % 8 != 0)
		++successorsMatrixSize;
	
	uint8_t *successorsMatrix = new uint8_t[successorsMatrixSize];
	memset(successorsMatrix, 0, successorsMatrixSize*sizeof(uint8_t));

	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		for (uint16_t j = 0; j < numberOfSuccessors[i]; ++j)	{
			uint16_t activityId = i;
			uint16_t successorId = activitiesSuccessors[i][j];
			uint32_t bitPossition = activityId*numberOfActivities+successorId;
			uint32_t bitIndex = bitPossition % 8;
			uint32_t byteIndex = bitPossition/8;
			successorsMatrix[byteIndex] |= (1<<bitIndex);
		}
	}

	// The longest path from the start activity to the end activity is computed.
	uint16_t *leftRightLongestPaths = computeLowerBounds(0);
	if (numberOfActivities > 0)
		criticalPathMakespan = leftRightLongestPaths[numberOfActivities-1];
	else
		criticalPathMakespan = -1;
	delete[] leftRightLongestPaths;

	/* THE TRANSFORMED LONGEST PATHS */

	/*
	 * It transformes the instance graph. Directions of edges are changed.
	 * The longest paths are computed from the end dummy activity to the others.
	 * After that the graph is transformed back.
	 */
	swap(numberOfSuccessors, numberOfPredecessors);
	swap(activitiesSuccessors, activitiesPredecessors);
	uint16_t *longestPaths = computeLowerBounds(numberOfActivities-1, true);
	swap(numberOfSuccessors, numberOfPredecessors);
	swap(activitiesSuccessors, activitiesPredecessors);
	
	/* CREATE INITIAL START SET SOLUTIONS */
	srand(time(NULL));
	SolutionInfo *infoAboutSchedules = new SolutionInfo[ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS];
	uint16_t *randomSchedules = new uint16_t[ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS*numberOfActivities], *schedWr = randomSchedules;

	uint32_t bestSetCost = UINT32_MAX;
	uint16_t *bestSetOrder = schedWr;
	uint16_t *startTimeValues = new uint16_t[numberOfActivities];

	for (uint16_t b = 0; b < ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS; ++b)	{
		makeDiversification(activitiesOrder, successorsMatrix, 100);
		uint16_t costOfSetSolution;
		if ((b % 2) == 0)	{
			costOfSetSolution = forwardScheduleEvaluation(activitiesOrder, startTimeValues);
		} else {
			costOfSetSolution = shakingDownEvaluation(activitiesOrder, startTimeValues);
			convertStartTimesById2ActivitiesOrder(activitiesOrder, startTimeValues);
		}
		infoAboutSchedules[b].readCounter = 0;
		infoAboutSchedules[b].solutionCost = costOfSetSolution;
		if (bestSetCost > costOfSetSolution)	{
			bestSetOrder = schedWr;
			bestSetCost = costOfSetSolution;
		}
		schedWr = copy(activitiesOrder, activitiesOrder+numberOfActivities, schedWr);
	}
	delete[] startTimeValues;

	/* CUDA INFO + DATA PHASE */

	bool cudaError = false;

	/* SET BASE CONFIG PARAMETERS */

	cudaData.numberOfActivities = numberOfActivities;
	cudaData.numberOfResources = numberOfResources;
	cudaData.maxTabuListSize = ConfigureRCPSP::TABU_LIST_SIZE;
	cudaData.solutionsSetSize = ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS;
	cudaData.swapRange = ConfigureRCPSP::SWAP_RANGE;
	cudaData.useTabuHash = ConfigureRCPSP::USE_TABU_HASH;
	cudaData.maximalValueOfReadCounter = ConfigureRCPSP::MAXIMAL_VALUE_OF_READ_COUNTER;
	cudaData.numberOfDiversificationSwaps = ConfigureRCPSP::DIVERSIFICATION_SWAPS;
	// Select evaluation algorithm.
	if (numberOfActivities < 100)	{
		cudaData.capacityResolutionAlgorithm = false;
	} else if (numberOfActivities >= 100 && numberOfActivities < 140)	{
		// It computes required parameters.
		double branchFactor;
		uint32_t sumOfCapacities = 0, sumOfSuccessors = 0;
		uint8_t minimalResourceCapacity, maximalResourceCapacity;
		for (uint32_t r = 0; r < numberOfResources; ++r)
			sumOfCapacities += capacityOfResources[r];
		for (uint32_t i = 0; i < numberOfActivities; ++i)
			sumOfSuccessors += numberOfSuccessors[i];
		minimalResourceCapacity = *min_element(capacityOfResources, capacityOfResources+numberOfResources);
		maximalResourceCapacity = *max_element(capacityOfResources, capacityOfResources+numberOfResources);
		branchFactor = (((double) sumOfSuccessors)/((double) numberOfActivities));
		// Decision what evaluation algorithm should be used.
		if (minimalResourceCapacity >= 29)
			cudaData.capacityResolutionAlgorithm = false;
		else if (sumOfCapacities >= 116 && branchFactor > 2.106)
			cudaData.capacityResolutionAlgorithm = false;
		else if (minimalResourceCapacity >= 25 && maximalResourceCapacity >= 42)
			cudaData.capacityResolutionAlgorithm = false;
		else
			cudaData.capacityResolutionAlgorithm = true;
	} else {
		cudaData.capacityResolutionAlgorithm = true;
	}
	cudaData.criticalPathLength = criticalPathMakespan;

	/* GET CUDA INFO - SET NUMBER OF THREADS PER BLOCK */

	int devId = 0;
	cudaDeviceProp prop;
	if (cudaGetDevice(&devId) == cudaSuccess && cudaGetDeviceProperties(&prop, devId) == cudaSuccess)	{
		if (verbose == true)	{
			cout<<"Device number: "<<devId<<endl;
			cout<<"Device name: "<<prop.name<<endl;
			cout<<"Device compute capability: "<<prop.major<<"."<<prop.minor<<endl;
			cout<<"Number of multiprocessors: "<<prop.multiProcessorCount<<endl;
			cout<<"Clock rate: "<<prop.clockRate<<endl;
			cout<<"Size of global memory: "<<prop.totalGlobalMem<<endl;
			cout<<"Size of shared memory per multiprocessor: "<<prop.sharedMemPerBlock<<endl;
			cout<<"Number of 32-bit registers per multiprocessor: "<<prop.regsPerBlock<<endl;
			cout<<"Size of constant memory: "<<prop.totalConstMem<<endl<<endl;
		}

		cudaCapability = prop.major*100+prop.minor*10;
		numberOfBlock = prop.multiProcessorCount*ConfigureRCPSP::NUMBER_OF_BLOCKS_PER_MULTIPROCESSOR;

		uint16_t sumOfCapacity = 0;
		for (uint8_t i = 0; i < numberOfResources; ++i)
			sumOfCapacity += capacityOfResources[i];

		cudaData.sumOfCapacities = sumOfCapacity;
		cudaData.maximalCapacityOfResource = *max_element(capacityOfResources, capacityOfResources+numberOfResources);
		numberOfThreadsPerBlock = 512;

		/* COMPUTE DYNAMIC MEMORY REQUIREMENT */
		dynSharedMemSize = numberOfThreadsPerBlock*sizeof(MoveInfo);	// merge array

		if ((numberOfActivities-2)*cudaData.swapRange < USHRT_MAX)
			dynSharedMemSize += numberOfThreadsPerBlock*sizeof(uint16_t); // merge help array
		else
			dynSharedMemSize += numberOfThreadsPerBlock*sizeof(uint32_t); // merge help array

		dynSharedMemSize += numberOfActivities*sizeof(uint16_t);	// block order
		dynSharedMemSize += (numberOfResources+1)*sizeof(uint16_t);	// resources indices
		dynSharedMemSize += numberOfActivities*sizeof(uint8_t);		// duration of activities

		if (cudaCapability < 200)	{
			cerr<<"Pre-Fermi cards aren't supported! Sorry..."<<endl;
			cudaError = true;
		}
	} else {
		cudaError = errorHandler(-1);
	}

	// If is possible to run 2 block (shared memory restriction) then copy successorsMatrix to shared memory.
	if (dynSharedMemSize+successorsMatrixSize*sizeof(uint8_t) < 7950)	{
		dynSharedMemSize += successorsMatrixSize*sizeof(uint8_t);
		cudaData.copySuccessorsMatrixToSharedMemory = true;
	} else	{
		cudaData.copySuccessorsMatrixToSharedMemory = false;
	}
	cudaData.successorsMatrixSize = successorsMatrixSize;

	if (verbose == true)	{
		cout<<"Dynamic shared memory requirement: "<<dynSharedMemSize<<endl;
		cout<<"Number of threads per block: "<<numberOfThreadsPerBlock<<endl<<endl;
	}

	/* COPY ACTIVITIES DURATION TO CUDA */
	if (!cudaError && cudaMalloc((void**) &cudaData.activitiesDuration, numberOfActivities*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(-1);
	}
	if (!cudaError && cudaMemcpy(cudaData.activitiesDuration, activitiesDuration, numberOfActivities*sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(0);
	}

	/* COPY PREDECESSORS+INDICES TO CUDA TEXTURE MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaPredecessorsArray, numOfElPred*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(0);
	}
	if (!cudaError && cudaMemcpy(cudaPredecessorsArray, linPredArray, numOfElPred*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(1);
	}
	if (!cudaError && bindTexture(cudaPredecessorsArray, numOfElPred, PREDECESSORS) != cudaSuccess)	{
		cudaError = errorHandler(1);
	}
	if (!cudaError && cudaMalloc((void**) &cudaPredecessorsIdxsArray, (numberOfActivities+1)*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(2);
	}
	if (!cudaError && cudaMemcpy(cudaPredecessorsIdxsArray, predIdxs, (numberOfActivities+1)*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(3);
	}
	if (!cudaError && bindTexture(cudaPredecessorsIdxsArray, numberOfActivities+1, PREDECESSORS_INDICES) != cudaSuccess)	{
		cudaError = errorHandler(3);
	}

	/* COPY SUCCESSORS BIT MATRIX */
	if (!cudaError && cudaMalloc((void**) &cudaData.successorsMatrix, successorsMatrixSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(4);
	}
	if (!cudaError && cudaMemcpy(cudaData.successorsMatrix, successorsMatrix, successorsMatrixSize*sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(5);
	}

	/* COPY ACTIVITIES RESOURCE REQUIREMENTS TO TEXTURE MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaActivitiesResourcesArray, resourceReqSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(5);
	}
	if (!cudaError && cudaMemcpy(cudaActivitiesResourcesArray, reqResLin, resourceReqSize*sizeof(uint8_t), cudaMemcpyHostToDevice)	!= cudaSuccess)	{
		cudaError = errorHandler(6);
	}
	if (!cudaError && bindTexture(cudaActivitiesResourcesArray, resourceReqSize, ACTIVITIES_RESOURCES) != cudaSuccess)	{
		cudaError = errorHandler(6);
	}

	/* COPY RESOURCES CAPACITIES TO CUDA MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaData.resourceIndices, (numberOfResources+1)*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(7);
	}
	if (!cudaError && cudaMemcpy(cudaData.resourceIndices, resIdxs, (numberOfResources+1)*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(8);
	}

	/* COPY TABU LISTS TO CUDA MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaData.tabuLists, numberOfBlock*cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(8);
	}
	if (!cudaError && cudaMemset(cudaData.tabuLists, 0, numberOfBlock*cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(9);
	}

	/* CREATE TABU CACHE */
	if (!cudaError && cudaMalloc((void**) &cudaData.tabuCaches, numberOfActivities*numberOfActivities*numberOfBlock*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(9);
	}
	if (!cudaError && cudaMemset(cudaData.tabuCaches, 0, numberOfActivities*numberOfActivities*numberOfBlock*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(10);
	}

	/* ALLOCATE AND INIT HASH MAP */
	if (!cudaError && cudaMalloc((void**) &cudaData.hashMap, HASH_TABLE_SIZE*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(10);
	}
	if (!cudaError && cudaMemset(cudaData.hashMap, 0, HASH_TABLE_SIZE*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(11);
	}

	/* COPY INITIAL SET SOLUTIONS */
	if (!cudaError && cudaMalloc((void**) &cudaData.solutionsSet, cudaData.solutionsSetSize*numberOfActivities*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(11);
	}
	if (!cudaError && cudaMemcpy(cudaData.solutionsSet, randomSchedules, cudaData.solutionsSetSize*numberOfActivities*sizeof(int16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(12);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.solutionsSetInfo, cudaData.solutionsSetSize*sizeof(SolutionInfo)) != cudaSuccess)	{
		cudaError = errorHandler(12);
	}
	if (!cudaError && cudaMemcpy(cudaData.solutionsSetInfo, infoAboutSchedules, cudaData.solutionsSetSize*sizeof(SolutionInfo), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(13);
	}

	/* CREATE TABU LISTS FOR SET OF SOLUTIONS */
	if (!cudaError && cudaMalloc((void**) &cudaData.solutionSetTabuLists, cudaData.solutionsSetSize*cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(13);
	}
	if (!cudaError && cudaMemset(cudaData.solutionSetTabuLists, 0, cudaData.solutionsSetSize*cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(14);
	}

	/* STATE OF COMMUNICATION FOR A SET OF SOLUTIONS */
	if (!cudaError && cudaMalloc((void**) &cudaData.lockSetSolution, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(14);
	}
	if (!cudaError && cudaMemset(cudaData.lockSetSolution, DATA_AVAILABLE, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(15);
	}

	/* GLOBAL BEST SOLUTION OF ALL BLOCKS AND COST OF THIS SOLUTION */
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolution, numberOfActivities*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(15);
	}
	if (!cudaError && cudaMemcpy(cudaData.globalBestSolution, bestSetOrder, numberOfActivities*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(16);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolutionCost, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(16);
	}
	if (!cudaError && cudaMemcpy(cudaData.globalBestSolutionCost, &bestSetCost, sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(17);
	}

	/* TABU LIST OF GLOBAL BEST SOLUTION */
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolutionTabuList, cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(17);
	}
	if (!cudaError && cudaMemset(cudaData.globalBestSolutionTabuList, 0, cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(18);
	}

	/* STATE OF COMMUNICATION FOR A GLOBAL BEST SOLUTION */
	if (!cudaError && cudaMalloc((void**) &cudaData.lockGlobalSolution, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(18);
	}
	if (!cudaError && cudaMemset(cudaData.lockGlobalSolution, DATA_AVAILABLE, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(19);
	}

	/* BEST CURRENT SOLUTIONS OF THE BLOCKS */
	if (!cudaError && cudaMalloc((void**) &cudaData.blocksBestSolution, numberOfActivities*numberOfBlock*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(19);
	}

	/* CREATE SWAP PENALTY FREE MERGE ARRAYS */
	if (!cudaError && cudaMalloc((void**) &cudaData.swapFreeMergeArray, (numberOfActivities-2)*cudaData.swapRange*numberOfBlock*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(20);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.mergeHelpArray, (numberOfActivities-2)*cudaData.swapRange*numberOfBlock*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(21);
	}

	/* CREATE COUNTER TO COUNT NUMBER OF EVALUATED SCHEDULES */
	if (!cudaError && cudaMalloc((void**) &cudaData.evaluatedSchedules, sizeof(uint64_t)) != cudaSuccess)	{
		cudaError = errorHandler(22);
	}
	if (!cudaError && cudaMemset(cudaData.evaluatedSchedules, 0, sizeof(uint64_t)) != cudaSuccess)	{
		cudaError = errorHandler(23);
	}

	/* COPY SUCCESSORS+INDICES TO CUDA TEXTURE MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaSuccessorsArray, numOfElSuc*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(23);
	}
	if (!cudaError && cudaMemcpy(cudaSuccessorsArray, linSucArray, numOfElSuc*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(24);
	}
	if (!cudaError && bindTexture(cudaSuccessorsArray, numOfElSuc, SUCCESSORS) != cudaSuccess)	{
		cudaError = errorHandler(24);
	}
	if (!cudaError && cudaMalloc((void**) &cudaSuccessorsIdxsArray, (numberOfActivities+1)*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(25);
	}
	if (!cudaError && cudaMemcpy(cudaSuccessorsIdxsArray, sucIdxs, (numberOfActivities+1)*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(26);
	}
	if (!cudaError && bindTexture(cudaSuccessorsIdxsArray, numberOfActivities+1, SUCCESSORS_INDICES) != cudaSuccess)	{
		cudaError = errorHandler(26);
	}
	
	/* COPY THE LONGEST PATH TO THE CONSTANT MEMORY */
	if (!cudaError && memcpyToSymbol((void*) longestPaths, numberOfActivities, THE_LONGEST_PATHS) != cudaSuccess)	{
		cudaError = errorHandler(27);
	}


	/* FREE ALLOCATED TEMPORARY RESOURCES */
	delete[] successorsMatrix;
	delete[] resIdxs;
	delete[] randomSchedules;
	delete[] infoAboutSchedules;
	delete[] longestPaths;
	delete[] reqResLin;
	delete[] linSucArray;
	delete[] linPredArray;
	delete[] sucIdxs;
	delete[] predIdxs;
	delete[] activitiesOrder;


	return cudaError;
}

bool ScheduleSolver::errorHandler(int16_t phase)	{
	cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
	switch (phase)	{
		case 27:
			unbindTexture(SUCCESSORS_INDICES);
		case 26:
			cudaFree(cudaSuccessorsIdxsArray);
		case 25:
			unbindTexture(SUCCESSORS);
		case 24:
			cudaFree(cudaSuccessorsArray);
		case 23:
			cudaFree(cudaData.evaluatedSchedules);
		case 22:
			cudaFree(cudaData.mergeHelpArray);
		case 21:
			cudaFree(cudaData.swapFreeMergeArray);
		case 20:
			cudaFree(cudaData.blocksBestSolution);
		case 19:
			cudaFree(cudaData.lockGlobalSolution);
		case 18:
			cudaFree(cudaData.globalBestSolutionTabuList);
		case 17:
			cudaFree(cudaData.globalBestSolutionCost);
		case 16:
			cudaFree(cudaData.globalBestSolution);
		case 15:
			cudaFree(cudaData.lockSetSolution);
		case 14:
			cudaFree(cudaData.solutionSetTabuLists);
		case 13:
			cudaFree(cudaData.solutionsSetInfo);
		case 12:
			cudaFree(cudaData.solutionsSet);
		case 11:
			cudaFree(cudaData.hashMap);
		case 10:
			cudaFree(cudaData.tabuCaches);
		case 9:
			cudaFree(cudaData.tabuLists);
		case 8:
			cudaFree(cudaData.resourceIndices);
		case 7:
			unbindTexture(ACTIVITIES_RESOURCES);
		case 6:
			cudaFree(cudaActivitiesResourcesArray);
		case 5:
			cudaFree(cudaData.successorsMatrix);
		case 4:
			unbindTexture(PREDECESSORS_INDICES);
		case 3:
			cudaFree(cudaPredecessorsIdxsArray);
		case 2:
			unbindTexture(PREDECESSORS);
		case 1:
			cudaFree(cudaPredecessorsArray);
		case 0:
			cudaFree(cudaData.activitiesDuration);

		default:
			break;
	}
	return true;
}

void ScheduleSolver::solveSchedule(const uint32_t& maxIter, const uint32_t& maxIterSinceBest)	{
	#ifdef __GNUC__
	timeval startTime, endTime, diffTime;
	gettimeofday(&startTime, NULL);
	#elif defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER startTimeStamp, stopTimeStamp;
	QueryPerformanceFrequency(&ticksPerSecond);
	QueryPerformanceCounter(&startTimeStamp);
	#endif

	// Set iterations parameters.
	cudaData.numberOfIterationsPerBlock = maxIter;
	cudaData.maximalIterationsSinceBest = maxIterSinceBest;

	/* RUN CUDA RCPSP SOLVER */

	runCudaSolveRCPSP(numberOfBlock, numberOfThreadsPerBlock, cudaCapability, dynSharedMemSize, cudaData);

	/* GET BEST FOUND SOLUTION */

	bool cudaError = false;
	uint32_t bestScheduleCost = 0;
	if (!cudaError && cudaMemcpy(&bestScheduleCost, cudaData.globalBestSolutionCost, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError && cudaMemcpy(bestScheduleOrder, cudaData.globalBestSolution, numberOfActivities*sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError && cudaMemcpy(&numberOfEvaluatedSchedules, cudaData.evaluatedSchedules, sizeof(uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError && bestScheduleCost < 0xffffffff)	{
		solutionComputed = true;
	}

	if (cudaError)	{
		throw runtime_error("ScheduleSolver::solveSchedule: Error occur when try to solve the instance!");
	}

	/* PRINT TABU HASH STATISTICS IF DEBUG MODE IS ON */
	#if DEBUG_TABU_HASH == 1
	uint32_t *hashMap = new uint32_t[HASH_TABLE_SIZE];
	if (cudaMemcpy(hashMap, cudaData.hashMap, sizeof(uint32_t)*HASH_TABLE_SIZE, cudaMemcpyDeviceToHost) == cudaSuccess)	{
		map<uint32_t,uint32_t> counters;
		for (uint32_t i = 0; i < HASH_TABLE_SIZE; ++i)  {
			counters[hashMap[i]]++;
		}

		uint64_t numberCollision = 0, numberCorrect = 0;
		for (map<uint32_t,uint32_t>::const_iterator it = counters.begin(); it != counters.end(); ++it)  {
			cout<<it->first<<": "<<it->second<<endl;
			if (it->first > 1)  {
				numberCollision += it->first*it->second;
			} else if (it->first == 1)  {
				numberCorrect = it->second;
			}
		}

		cout<<"Number of permutation: "<<numberCollision+numberCorrect<<endl;
		cout<<"Correct/collision permutation: "<<numberCorrect<<"/"<<numberCollision<<endl;
		cout<<"Total accuracy: "<<numberCorrect/((double) numberCollision+numberCorrect)<<endl<<endl;
	} else {
		cerr<<"Cannot copy tabu hash from device memory!"<<endl<<endl;;
	}
	delete[] hashMap;
	#endif

	#ifdef __GNUC__
	gettimeofday(&endTime, NULL);
	timersub(&endTime, &startTime, &diffTime);
	totalRunTime = diffTime.tv_sec+diffTime.tv_usec/1000000.;
	#elif defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
	QueryPerformanceCounter(&stopTimeStamp);
	totalRunTime = (stopTimeStamp.QuadPart-startTimeStamp.QuadPart)/((double) ticksPerSecond.QuadPart);
	#endif
}

void ScheduleSolver::printBestSchedule(bool verbose, ostream& output)	const	{
	printSchedule(bestScheduleOrder, verbose, output);
}

void ScheduleSolver::writeBestScheduleToFile(const string& fileName) {
	ofstream out(fileName.c_str(), ios::out | ios::binary | ios::trunc);

	/* WRITE INTANCE DATA */
	uint32_t numberOfActivitiesUint32 = numberOfActivities, numberOfResourcesUint32 = numberOfResources;
	out.write((const char*) &numberOfActivitiesUint32, sizeof(uint32_t));
	out.write((const char*) &numberOfResourcesUint32, sizeof(uint32_t));

	uint32_t *activitiesDurationUint32 = convertArrayType<uint8_t, uint32_t>(activitiesDuration, numberOfActivities);
	out.write((const char*) activitiesDurationUint32, numberOfActivities*sizeof(uint32_t));
	uint32_t *capacityOfResourcesUint32 = convertArrayType<uint8_t, uint32_t>(capacityOfResources, numberOfResources);
	out.write((const char*) capacityOfResourcesUint32, numberOfResources*sizeof(uint32_t));
	for (uint32_t i = 0; i < numberOfActivities; ++i)	{
		uint32_t *activityRequiredResourcesUint32 = convertArrayType<uint8_t, uint32_t>(activitiesResources[i], numberOfResources);
		out.write((const char*) activityRequiredResourcesUint32, numberOfResources*sizeof(uint32_t));
		delete[] activityRequiredResourcesUint32;
	}

	uint32_t *numberOfSuccessorsUint32 = convertArrayType<uint16_t, uint32_t>(numberOfSuccessors, numberOfActivities);
	out.write((const char*) numberOfSuccessorsUint32, numberOfActivities*sizeof(uint32_t));
	for (uint32_t i = 0; i < numberOfActivities; ++i)	{
		uint32_t *activitySuccessorsUint32 = convertArrayType<uint16_t, uint32_t>(activitiesSuccessors[i], numberOfSuccessors[i]);
		out.write((const char*) activitySuccessorsUint32, numberOfSuccessors[i]*sizeof(uint32_t));
		delete[] activitySuccessorsUint32;
	}

	uint32_t *numberOfPredecessorsUint32 = convertArrayType<uint16_t, uint32_t>(numberOfPredecessors, numberOfActivities);
	out.write((const char*) numberOfPredecessorsUint32, numberOfActivities*sizeof(uint32_t));
	for (uint32_t i = 0; i < numberOfActivities; ++i)	{
		uint32_t *activityPredecessorsUint32 = convertArrayType<uint16_t, uint32_t>(activitiesPredecessors[i], numberOfPredecessors[i]);
		out.write((const char*) activityPredecessorsUint32, numberOfPredecessors[i]*sizeof(uint32_t));
		delete[] activityPredecessorsUint32;
	}

	delete[] numberOfPredecessorsUint32;
	delete[] numberOfSuccessorsUint32;
	delete[] capacityOfResourcesUint32;
	delete[] activitiesDurationUint32;

	/* WRITE RESULTS */
	uint16_t *startTimesById = new uint16_t[numberOfActivities], *copyOrder = new uint16_t[numberOfActivities];
	uint32_t scheduleLength = shakingDownEvaluation(bestScheduleOrder, startTimesById);

	uint16_t *copyWr = copyOrder;
	copy(bestScheduleOrder, bestScheduleOrder+numberOfActivities, copyWr);
	convertStartTimesById2ActivitiesOrder(copyOrder, startTimesById);

	out.write((const char*) &scheduleLength, sizeof(uint32_t));
	uint32_t *copyOrderUint32 = convertArrayType<uint16_t, uint32_t>(copyOrder, numberOfActivities);
	out.write((const char*) copyOrderUint32, numberOfActivities*sizeof(uint32_t));
	uint32_t *startTimesByIdUint32 = convertArrayType<uint16_t, uint32_t>(startTimesById, numberOfActivities);
	out.write((const char*) startTimesByIdUint32, numberOfActivities*sizeof(uint32_t));

	delete[] startTimesByIdUint32;
	delete[] copyOrderUint32;
	delete[] startTimesById;
	delete[] copyOrder;

	out.close();
}

uint16_t* ScheduleSolver::computeLowerBounds(const uint16_t& startActivityId, const bool& energyReasoning) const {
	// The first dummy activity is added to list.
	list<uint16_t> expandedNodes(1, startActivityId);
	// We have to remember closed activities. (the bound of the activity is determined)
	bool *closedActivities = new bool[numberOfActivities];
	fill(closedActivities, closedActivities+numberOfActivities, false);
	// The longest path from the start activity to the activity at index "i".
	uint16_t *maxDistances = new uint16_t[numberOfActivities];
	fill(maxDistances, maxDistances+numberOfActivities, 0);
	// All branches that go through nodes are saved.
	// branches[i][j] = p -> The p-nd branch that started in the node j goes through node i.
	map<uint16_t, uint16_t> * branches = new map<uint16_t, uint16_t>[numberOfActivities];

	while (!expandedNodes.empty())	{
		uint16_t activityId;
		uint16_t minimalStartTime;
		// We select the first activity with all predecessors closed.
		list<uint16_t>::iterator lit = expandedNodes.begin();
		while (lit != expandedNodes.end())	{
			activityId = *lit;
			if (closedActivities[activityId] == false)	{
				minimalStartTime = 0;
				bool allPredecessorsClosed = true;
				vector<map<uint16_t, uint16_t> > predecessorsBranches;
				uint16_t *activityPredecessors = activitiesPredecessors[activityId];
				for (uint16_t* p = activityPredecessors; p < activityPredecessors+numberOfPredecessors[activityId]; ++p)	{
					if (closedActivities[*p] == false)	{
						allPredecessorsClosed = false;
						break;
					} else {
						// It updates the maximal distance from the start activity to the activity "activityId".
						minimalStartTime = max(maxDistances[*p]+activitiesDuration[*p], minimalStartTime);
						if (numberOfPredecessors[activityId] > 1 && energyReasoning)
							predecessorsBranches.push_back(branches[*p]);
					}
				}
				if (allPredecessorsClosed)	{
					if (numberOfPredecessors[activityId] > 1 && energyReasoning) {
						// Output branches are found out for the node with more predecessors.
						map<uint16_t, uint16_t> newBranches;
						set<uint16_t> startNodesOfMultiPaths;
						for (uint32_t k = 0; k < predecessorsBranches.size(); ++k)	{
							map<uint16_t, uint16_t>& m = predecessorsBranches[k];
							for (map<uint16_t, uint16_t>::const_iterator mit = m.begin(); mit != m.end(); ++mit)	{
								map<uint16_t, uint16_t>::const_iterator sit;
								if ((sit = newBranches.find(mit->first)) == newBranches.end())	{
									newBranches[mit->first] = mit->second;
								} else {
									// The branch number has to be checked.
									if (mit->second != sit->second)	{
										// Multi-paths were detected! New start node is stored.
										startNodesOfMultiPaths.insert(mit->first);
									}
								}
							}
						}
						branches[activityId] = newBranches;
						// If more than one path exists to the node "activityId", then the resource restrictions
						// are taken into accout to improve lower bound.
						uint16_t minimalResourceStartTime = 0;
						for (set<uint16_t>::const_iterator sit = startNodesOfMultiPaths.begin(); sit != startNodesOfMultiPaths.end(); ++sit)	{
							// Vectors are sorted by activity id's.
							vector<uint16_t> allSuccessors = getAllActivitySuccessors(*sit);
							vector<uint16_t> allPredecessors = getAllActivityPredecessors(activityId);
							// The vector of all activities between the activity "i" and activity "j".
							vector<uint16_t> intersectionOfActivities;
							set_intersection(allPredecessors.begin(), allPredecessors.end(), allSuccessors.begin(),
									allSuccessors.end(), back_inserter(intersectionOfActivities));
							for (uint8_t k = 0; k < numberOfResources; ++k)	{
								uint32_t sumOfEnergy = 0, timeInterval;
								for (uint16_t i = 0; i < intersectionOfActivities.size(); ++i)	{
									uint16_t innerActivityId = intersectionOfActivities[i];
									sumOfEnergy += activitiesDuration[innerActivityId]*activitiesResources[innerActivityId][k];
								}

								timeInterval = sumOfEnergy/capacityOfResources[k];
								if ((sumOfEnergy % capacityOfResources[k]) != 0)
									++timeInterval;
								
								minimalResourceStartTime = max(minimalResourceStartTime, 
										maxDistances[*sit]+activitiesDuration[*sit]+timeInterval); 
							}
						}
						minimalStartTime = max(minimalStartTime, minimalResourceStartTime);
					}
					break;
				}

				++lit;
			} else {
				lit = expandedNodes.erase(lit);
			}
		}
		
		if (lit != expandedNodes.end())	{
			closedActivities[activityId] = true;
			maxDistances[activityId] = minimalStartTime;
			expandedNodes.erase(lit);
			uint32_t numberOfSuccessorsOfClosedActivity = numberOfSuccessors[activityId];
			for (uint32_t s = 0; s < numberOfSuccessorsOfClosedActivity; ++s)	{
				uint32_t successorId = activitiesSuccessors[activityId][s];
				if (numberOfPredecessors[successorId] <= 1 && energyReasoning)	{
					branches[successorId] = branches[activityId];
					if (numberOfSuccessorsOfClosedActivity > 1)	{
						branches[successorId][activityId] = s;
					}
				}
				expandedNodes.push_back(successorId);
			}
		} else {
			break;
		}
	}

	delete[] branches;
	delete[] closedActivities; 

	return maxDistances;
}

uint16_t ScheduleSolver::evaluateOrder(const uint16_t * const& order, const uint16_t * const * const& relatedActivities,
	       	const uint16_t * const& numberOfRelatedActivities, uint16_t *& timeValuesById, bool forwardEvaluation) const {
	SourcesLoad sourcesLoad(numberOfResources, capacityOfResources, upperBoundMakespan);

	uint32_t scheduleLength = 0;
	for (uint32_t i = 0; i < numberOfActivities; ++i)	{
		uint32_t start = 0;
		uint32_t activityId = order[forwardEvaluation == true ? i : numberOfActivities-i-1];
		for (uint32_t j = 0; j < numberOfRelatedActivities[activityId]; ++j)	{
			uint32_t relatedActivityId = relatedActivities[activityId][j];
			start = max(timeValuesById[relatedActivityId]+activitiesDuration[relatedActivityId], start);
		}

		start = max(sourcesLoad.getEarliestStartTime(activitiesResources[activityId], start, activitiesDuration[activityId]), start);
		sourcesLoad.addActivity(start, start+activitiesDuration[activityId], activitiesResources[activityId]);
		scheduleLength = max(scheduleLength, start+activitiesDuration[activityId]);

		timeValuesById[activityId] = start;
	}

	return scheduleLength;
}

uint16_t ScheduleSolver::forwardScheduleEvaluation(const uint16_t * const& order, uint16_t *& startTimesById) const {
	return evaluateOrder(order, activitiesPredecessors, numberOfPredecessors, startTimesById, true);
}

uint16_t ScheduleSolver::backwardScheduleEvaluation(const uint16_t * const& order, uint16_t *& startTimesById) const {
	uint16_t makespan = evaluateOrder(order, activitiesSuccessors, numberOfSuccessors, startTimesById, false);
	// It computes the latest start time value for each activity.
	for (uint16_t id = 0; id < numberOfActivities; ++id)
		startTimesById[id] = makespan-startTimesById[id]-activitiesDuration[id];
	return makespan;
}

uint16_t ScheduleSolver::shakingDownEvaluation(const uint16_t * const& order, uint16_t *bestScheduleStartTimesById) const {
	int32_t scheduleLength = 0;
	uint16_t bestScheduleLength = 0xffff;
	uint16_t *currentOrder = new uint16_t[numberOfActivities];
	uint16_t *timeValuesById = new uint16_t[numberOfActivities];

	for (uint16_t i = 0; i < numberOfActivities; ++i)
		currentOrder[i] = order[i];

	while (true)	{
		// Forward schedule...
		scheduleLength = forwardScheduleEvaluation(currentOrder, timeValuesById);
		if (scheduleLength < bestScheduleLength)	{
			bestScheduleLength = scheduleLength;
			if (bestScheduleStartTimesById != NULL)	{
				for (uint16_t id = 0; id < numberOfActivities; ++id)
					bestScheduleStartTimesById[id] = timeValuesById[id];
			}
		} else	{
			// No additional improvement can be found...
			break;
		}

		// It computes the earliest activities finish time.
		for (uint16_t id = 0; id < numberOfActivities; ++id)
			timeValuesById[id] += activitiesDuration[id];

		// Sort for backward phase..
		insertSort(currentOrder, timeValuesById, numberOfActivities);

		// Backward phase.
		int32_t scheduleLengthBackward = backwardScheduleEvaluation(currentOrder, timeValuesById);
		int32_t diffCmax = scheduleLength-scheduleLengthBackward;

		// It computes the latest start time of activities.
		for (uint16_t id = 0; id < numberOfActivities; ++id)	{
			if (((int32_t) timeValuesById[id])+diffCmax > 0)
				timeValuesById[id] += diffCmax;
			else
				timeValuesById[id] = 0;
		}

		// Sort for forward phase..
		insertSort(currentOrder, timeValuesById, numberOfActivities);
	}

	delete[] currentOrder;
	delete[] timeValuesById;

	return bestScheduleLength;
}

uint32_t ScheduleSolver::computePrecedencePenalty(const uint16_t * const& startTimesById)	const	{
	uint32_t penalty = 0;
	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		for (uint16_t j = 0; j < numberOfSuccessors[activityId]; ++j)	{
			uint16_t successorId = activitiesSuccessors[activityId][j];	
			if (startTimesById[activityId]+activitiesDuration[activityId] > startTimesById[successorId])
				penalty += startTimesById[activityId]+activitiesDuration[activityId]-startTimesById[successorId];
		}
	}
	return penalty;
}

bool ScheduleSolver::checkSwapPrecedencePenalty(const uint16_t * const& order, const uint8_t * const& successorsMatrix, uint16_t i, uint16_t j) const	{
	if (i > j) swap(i,j);
	for (uint32_t k = i; k < j; ++k)	{
		uint32_t bitPossition = order[k]*numberOfActivities+order[j];
		if ((successorsMatrix[bitPossition/8] & (1<<(bitPossition%8))) > 0)	{
			return false;
		}
	}
	for (uint32_t k = i+1; k <= j; ++k)	{
		uint32_t bitPossition = order[i]*numberOfActivities+order[k];
		if ((successorsMatrix[bitPossition/8] & (1<<(bitPossition%8))) > 0)	{
			return false;
		}
	}
	return true;
}

void ScheduleSolver::printSchedule(const uint16_t * const& scheduleOrder, bool verbose, ostream& output)	const	{
	if (solutionComputed == true)	{
		uint16_t *startTimesById = new uint16_t[numberOfActivities];
		uint16_t scheduleLength = shakingDownEvaluation(scheduleOrder, startTimesById);
		uint16_t precedencePenalty = computePrecedencePenalty(startTimesById);
	
		if (verbose == true)	{
			output<<"start\tactivities"<<endl;
			for (uint16_t c = 0; c <= scheduleLength; ++c)	{
				bool first = true;
				for (uint16_t id = 0; id < numberOfActivities; ++id)	{
					if (startTimesById[id] == c)	{
						if (first == true)	{
							output<<c<<":\t"<<id;
							first = false;
						} else {
							output<<" "<<id;
						}
					}
				}
				if (!first)	output<<endl;
			}
			output<<"Schedule length: "<<scheduleLength<<endl;
			output<<"Precedence penalty: "<<precedencePenalty<<endl;
			output<<"Critical path makespan: "<<criticalPathMakespan<<endl;
			output<<"Schedule solve time: "<<totalRunTime<<" s"<<endl;
			output<<"Total number of evaluated schedules: "<<numberOfEvaluatedSchedules<<endl;
		}	else	{
			output<<scheduleLength<<"+"<<precedencePenalty<<" "<<criticalPathMakespan<<"\t["<<totalRunTime<<" s]\t"<<numberOfEvaluatedSchedules<<endl;
		} 
		delete[] startTimesById;
	} else {
		output<<"Solution hasn't been computed yet!"<<endl;
	}
}

void ScheduleSolver::convertStartTimesById2ActivitiesOrder(uint16_t *order, const uint16_t * const& startTimesById) const {
	insertSort(order, startTimesById, numberOfActivities);
}

void ScheduleSolver::insertSort(uint16_t* order, const uint16_t * const& timeValuesById, const int32_t& size) {
	for (int32_t i = 1; i < size; ++i)	{
		for (int32_t j = i; (j > 0) && ((timeValuesById[order[j]] < timeValuesById[order[j-1]]) == true); --j)	{
			swap(order[j], order[j-1]);
		}
	}
}

void ScheduleSolver::makeDiversification(uint16_t * const& order, const uint8_t * const& successorsMatrix, const uint32_t& numberOfSwaps)	{
	uint32_t performedSwaps = 0;
	while (performedSwaps < numberOfSwaps)  {
		uint16_t i = (rand() % (numberOfActivities-2)) + 1;
		uint16_t j = (rand() % (numberOfActivities-2)) + 1;

		if ((i != j) && (checkSwapPrecedencePenalty(order, successorsMatrix, i, j) == true)) { 
			swap(order[i], order[j]);
			++performedSwaps;
		}
	}
}

vector<uint16_t> ScheduleSolver::getAllRelatedActivities(uint16_t activityId, uint16_t *numberOfRelated, uint16_t **related) const	{
	vector<uint16_t> relatedActivities;
	bool *activitiesSet = new bool[numberOfActivities];
	fill(activitiesSet, activitiesSet+numberOfActivities, false);

	for (uint16_t j = 0; j < numberOfRelated[activityId]; ++j)	{
		activitiesSet[related[activityId][j]] = true;
		vector<uint16_t> indirectRelated = getAllRelatedActivities(related[activityId][j], numberOfRelated, related);
		for (vector<uint16_t>::const_iterator it = indirectRelated.begin(); it != indirectRelated.end(); ++it)
			activitiesSet[*it] = true;
	}

	for (uint16_t id = 0; id < numberOfActivities; ++id)	{
		if (activitiesSet[id] == true)
			relatedActivities.push_back(id);
	}
	
	delete[] activitiesSet;
	return relatedActivities;
}

vector<uint16_t> ScheduleSolver::getAllActivitySuccessors(const uint16_t& activityId) const	{
	return getAllRelatedActivities(activityId, numberOfSuccessors, activitiesSuccessors);
}

vector<uint16_t> ScheduleSolver::getAllActivityPredecessors(const uint16_t& activityId) const 	{
	return getAllRelatedActivities(activityId, numberOfPredecessors, activitiesPredecessors);
}

template <class X, class Y>
Y* ScheduleSolver::convertArrayType(X* array, size_t length)	{
	Y* convertedArray = new Y[length];
	for (uint32_t i = 0; i < length; ++i)
		convertedArray[i] = array[i];
	return convertedArray;
}

void ScheduleSolver::freeCudaMemory()	{
	for (int i = 0; i < 5; ++i)
		unbindTexture(i);
	cudaFree(cudaSuccessorsIdxsArray);
	cudaFree(cudaSuccessorsArray);
	cudaFree(cudaData.activitiesDuration);
	cudaFree(cudaPredecessorsArray);
	cudaFree(cudaPredecessorsIdxsArray);
	cudaFree(cudaData.successorsMatrix);
	cudaFree(cudaActivitiesResourcesArray);
	cudaFree(cudaData.resourceIndices);
	cudaFree(cudaData.tabuLists);
	cudaFree(cudaData.tabuCaches);
	cudaFree(cudaData.hashMap);
	cudaFree(cudaData.solutionsSet);
	cudaFree(cudaData.solutionsSetInfo);
	cudaFree(cudaData.solutionSetTabuLists);
	cudaFree(cudaData.lockSetSolution);
	cudaFree(cudaData.globalBestSolution);
	cudaFree(cudaData.globalBestSolutionCost);
	cudaFree(cudaData.globalBestSolutionTabuList);
	cudaFree(cudaData.lockGlobalSolution);
	cudaFree(cudaData.blocksBestSolution);
	cudaFree(cudaData.swapFreeMergeArray);
	cudaFree(cudaData.mergeHelpArray);
	cudaFree(cudaData.evaluatedSchedules);
}

ScheduleSolver::~ScheduleSolver()	{
	freeCudaMemory();
	for (uint16_t actId = 0; actId < numberOfActivities; ++actId)	{
		delete[] activitiesPredecessors[actId];
	}
	delete[] activitiesPredecessors;
	delete[] numberOfPredecessors;
	delete[] bestScheduleOrder;
}



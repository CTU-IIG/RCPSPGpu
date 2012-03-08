#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <stdexcept>

#ifdef __GNUC__
#include <sys/time.h>
#elif defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
#include <Windows.h>
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

	/* CONVERT PREDECESSOR ARRAY TO 1D */
	uint32_t numOfElPred = 0;
	uint16_t *predIdxs = new uint16_t[numberOfActivities+1];

	predIdxs[0] = 0;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		numOfElPred += numberOfPredecessors[i];
		predIdxs[i+1] = numOfElPred;
	}

	uint16_t *linPredArray = new uint16_t[numOfElPred], *wrPtr = linPredArray;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		for (uint16_t j = 0; j < numberOfPredecessors[i]; ++j)	{
			*(wrPtr++) = activitiesPredecessors[i][j];
		}
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
	int32_t **distanceMatrix = new int32_t*[numberOfActivities];
	for (uint32_t i = 0; i < numberOfActivities; ++i)	{
		distanceMatrix[i] = new int32_t[numberOfActivities];
		memset(distanceMatrix[i], -1, sizeof(int32_t)*numberOfActivities);
	}

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
			uint32_t bitIndex = bitPossition%8;
			uint32_t byteIndex = bitPossition/8;
			successorsMatrix[byteIndex] |= (1<<bitIndex);
			distanceMatrix[activityId][successorId] = activitiesDuration[activityId]; 
		}
	}

	for (uint32_t i = 0; i < numberOfActivities; ++i)	{
		for (uint32_t j = 0; j < numberOfActivities; ++j)	{
			for (uint32_t k = 0; k < numberOfActivities; ++k)	{
				if (distanceMatrix[i][k] != -1 && distanceMatrix[k][j] != -1)	{
					if (distanceMatrix[i][j] != -1)	{
						if (distanceMatrix[i][j] < distanceMatrix[i][k]+distanceMatrix[k][j]) 
							distanceMatrix[i][j] = distanceMatrix[i][k]+distanceMatrix[k][j]; 
					} else {
						distanceMatrix[i][j] = distanceMatrix[i][k]+distanceMatrix[k][j]; 
					}
				}
			}
		}
	}

	if (numberOfActivities > 0)
		criticalPathMakespan = distanceMatrix[0][numberOfActivities-1];
	else
		criticalPathMakespan = -1;
	
	/* CREATE INITIAL START SET SOLUTIONS */
	srand(time(NULL));
	SolutionInfo *infoAboutSchedules = new SolutionInfo[ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS];
	uint16_t *randomSchedules = new uint16_t[ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS*numberOfActivities], *schedWr = randomSchedules;

	// Rewrite to sets.
	for (uint16_t b = 0; b < ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS; ++b)	{
		makeDiversification(activitiesOrder, successorsMatrix, 100);
		infoAboutSchedules[b].readCounter = 0;
		infoAboutSchedules[b].solutionCost = evaluateOrder(activitiesOrder);
		schedWr = copy(activitiesOrder, activitiesOrder+numberOfActivities, schedWr);
	}

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

		if (cudaCapability >= 200)	{
			uint16_t sumOfCapacity = 0;
			for (uint8_t i = 0; i < numberOfResources; ++i)
				sumOfCapacity += capacityOfResources[i];

			cudaData.sumOfCapacities = sumOfCapacity;
			cudaData.maximalCapacityOfResource = *max_element(capacityOfResources, capacityOfResources+numberOfResources);

			numberOfThreadsPerBlock = 512;
		}	else	{
			numberOfThreadsPerBlock = 256;
		}

		/* COMPUTE DYNAMIC MEMORY REQUIREMENT */
		dynSharedMemSize = numberOfThreadsPerBlock*sizeof(MoveInfo);	// merge array

		if ((numberOfActivities-2)*cudaData.swapRange < USHRT_MAX)
			dynSharedMemSize += numberOfThreadsPerBlock*sizeof(uint16_t); // merge help array
		else
			dynSharedMemSize += numberOfThreadsPerBlock*sizeof(uint32_t); // merge help array

		dynSharedMemSize += numberOfActivities*sizeof(uint16_t);	// block order
		dynSharedMemSize += (numberOfResources+1)*sizeof(uint16_t);	// resources indices
		dynSharedMemSize += numberOfActivities*sizeof(uint8_t);		// duration of activities
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
	if (!cudaError && cudaMalloc((void**) &cudaData.setStateOfCommunication, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(14);
	}
	if (!cudaError && cudaMemset(cudaData.setStateOfCommunication, DATA_AVAILABLE, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(15);
	}

	/* GLOBAL BEST SOLUTION OF ALL BLOCKS AND COST OF THIS SOLUTION */
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolution, numberOfActivities*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(15);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolutionCost, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(16);
	}
	if (!cudaError && cudaMemset(cudaData.globalBestSolutionCost, UCHAR_MAX, sizeof(uint32_t)) != cudaSuccess)	{
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
	if (!cudaError && cudaMalloc((void**) &cudaData.globalStateOfCommunication, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(18);
	}
	if (!cudaError && cudaMemset(cudaData.globalStateOfCommunication, DATA_AVAILABLE, sizeof(uint32_t)) != cudaSuccess)	{
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


	/* FREE ALLOCATED TEMPORARY RESOURCES */
	for (uint32_t i = 0; i < numberOfActivities; ++i)
		delete[] distanceMatrix[i];
	delete[] distanceMatrix;
	delete[] successorsMatrix;
	delete[] resIdxs;
	delete[] randomSchedules;
	delete[] infoAboutSchedules;
	delete[] reqResLin;
	delete[] linPredArray;
	delete[] predIdxs;
	delete[] activitiesOrder;

	return cudaError;
}

bool ScheduleSolver::errorHandler(int16_t phase)	{
	cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
	switch (phase)	{
		case 21:
			cudaFree(cudaData.swapFreeMergeArray);
		case 20:
			cudaFree(cudaData.blocksBestSolution);
		case 19:
			cudaFree(cudaData.globalStateOfCommunication);
		case 18:
			cudaFree(cudaData.globalBestSolutionTabuList);
		case 17:
			cudaFree(cudaData.globalBestSolutionCost);
		case 16:
			cudaFree(cudaData.globalBestSolution);
		case 15:
			cudaFree(cudaData.setStateOfCommunication);
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
	uint32_t bestGlobalCost;
	if (!cudaError && cudaMemcpy(&bestGlobalCost, cudaData.globalBestSolutionCost, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError && cudaMemcpy(bestScheduleOrder, cudaData.globalBestSolution, numberOfActivities*sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError && bestGlobalCost < 0xffffffff)	{
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

uint16_t ScheduleSolver::evaluateOrder(const uint16_t * const& order, uint16_t *startTimesWriter, uint16_t *startTimesWriterById)	const	{
	bool freeMem = false;
	uint16_t start = 0, scheduleLength = 0;
	SourcesLoad load(numberOfResources,capacityOfResources);
	if (startTimesWriterById == NULL)	{
		startTimesWriterById = new uint16_t[numberOfActivities];
		freeMem = true;
	}
	memset(startTimesWriterById, 0, sizeof(uint16_t)*numberOfActivities);

	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		uint16_t activityId = order[i];
		for (uint16_t j = 0; j < numberOfPredecessors[activityId]; ++j)	{
			uint16_t predecessorId = activitiesPredecessors[activityId][j];
			start = max(startTimesWriterById[predecessorId]+activitiesDuration[predecessorId], start);
		}

		start = max(load.getEarliestStartTime(activitiesResources[activityId]), start);
		load.addActivity(start, start+activitiesDuration[activityId], activitiesResources[activityId]);
		scheduleLength = max(scheduleLength, start+activitiesDuration[activityId]);

		if (startTimesWriter != NULL)
			*(startTimesWriter++) = start;

		startTimesWriterById[activityId] = start;
	}

	if (freeMem == true)
		delete[] startTimesWriterById; 
	return scheduleLength;
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

void ScheduleSolver::printSchedule(const uint16_t * const& scheduleOrder, bool verbose, ostream& OUT)	const	{
	if (solutionComputed == true)	{
		uint16_t *startTimes = new uint16_t[numberOfActivities];
		uint16_t *startTimesById = new uint16_t[numberOfActivities];

		size_t scheduleLength = evaluateOrder(scheduleOrder, startTimes, startTimesById);
		size_t precedencePenalty = computePrecedencePenalty(startTimesById);

		if (verbose == true)	{
			int32_t startTime = -1;
			OUT<<"start\tactivities"<<endl;
			for (uint16_t i = 0; i < numberOfActivities; ++i)	{
				if (startTime != ((int32_t) startTimes[i]))	{
					if (i != 0) OUT<<endl;
					OUT<<startTimes[i]<<":\t"<<(scheduleOrder[i]+1);
					startTime = startTimes[i];
				} else {
					OUT<<" "<<(scheduleOrder[i]+1);
				}
			}
			OUT<<endl;
			OUT<<"Schedule length: "<<scheduleLength<<endl;
			OUT<<"Precedence penalty: "<<precedencePenalty<<endl;
			OUT<<"Critical path makespan: "<<criticalPathMakespan<<endl; 
			OUT<<"Schedule solve time: "<<totalRunTime<<" s"<<endl;
		} else {
			OUT<<scheduleLength<<"+"<<precedencePenalty<<" "<<criticalPathMakespan<<"\t["<<totalRunTime<<" s]"<<endl; 
		}

		delete[] startTimesById;
		delete[] startTimes;
	} else {
		OUT<<"Solution hasn't been computed yet!"<<endl;
	}
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

void ScheduleSolver::printBestSchedule(bool verbose, ostream& OUT)	const	{
	printSchedule(bestScheduleOrder, verbose, OUT);
}

void ScheduleSolver::freeCudaMemory()	{
	for (int i = 0; i < 3; ++i)
		unbindTexture(i);
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
	cudaFree(cudaData.setStateOfCommunication);
	cudaFree(cudaData.globalBestSolution);
	cudaFree(cudaData.globalBestSolutionCost);
	cudaFree(cudaData.globalBestSolutionTabuList);
	cudaFree(cudaData.globalStateOfCommunication);
	cudaFree(cudaData.blocksBestSolution);
	cudaFree(cudaData.swapFreeMergeArray);
	cudaFree(cudaData.mergeHelpArray);
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



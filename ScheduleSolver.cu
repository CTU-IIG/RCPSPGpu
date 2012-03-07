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

using namespace std;

ScheduleSolver::ScheduleSolver(uint8_t resNum, uint8_t *capRes, uint16_t actNum, uint8_t *actDur, uint16_t **actSuc, uint16_t *actNumSuc, uint8_t **actRes, bool verbose)
		: numberOfResources(resNum), capacityOfResources(capRes), numberOfActivities(actNum), activitiesDuration(actDur),
		  activitiesSuccessors(actSuc), numberOfSuccessors(actNumSuc), activitiesResources(actRes), solutionComputed(false), totalRunTime(-1)	{

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
			cout<<"All required resources allocated..."<<endl;
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

	delete[] currentLevel;
	delete[] newCurrentLevel;
}

bool ScheduleSolver::prepareCudaMemory(uint16_t *activitiesOrder, bool verbose)	{

	bool cudaError = false;

	cudaData.numberOfActivities = numberOfActivities;
	cudaData.numberOfResources = numberOfResources;
	cudaData.maxTabuListSize = TABU_LIST_SIZE;
	cudaData.solutionsSetSize = SOLUTION_SET_SIZE;
	cudaData.swapRange = SWAP_RANGE;

	/* CONVERT SUCCESSOR AND PREDECESSOR ARRAYS TO 1D */
	size_t numOfElSuc = 0, numOfElPred = 0;
	uint16_t *sucIdxs = new uint16_t[numberOfActivities+1], *predIdxs = new uint16_t[numberOfActivities+1];

	sucIdxs[0] = predIdxs[0] = 0;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		numOfElSuc += numberOfSuccessors[i];
		numOfElPred += numberOfPredecessors[i];
		sucIdxs[i+1] = numOfElSuc;
		predIdxs[i+1] = numOfElPred;
	}

	uint16_t *linSucArray = new uint16_t[numOfElSuc], *linPredArray = new uint16_t[numOfElPred], *wrPtr1 = linSucArray, *wrPtr2 = linPredArray;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		for (uint16_t j = 0; j < numberOfSuccessors[i]; ++j)	{
			*(wrPtr1++) = activitiesSuccessors[i][j];	
		}
		for (uint16_t j = 0; j < numberOfPredecessors[i]; ++j)	{
			*(wrPtr2++) = activitiesPredecessors[i][j];
		}
	}

	/* GET CUDA INFO - SET NUMBER OF THREADS PER BLOCK */
	int devId = 0;
	cudaDeviceProp prop;
	uint32_t numberOfMultiprocessor = 0;
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

		dynSharedMemSize = 0;
		numberOfThreadsPerBlock = 0;
		numberOfMultiprocessor = prop.multiProcessorCount;
		cudaCapability = prop.major*100+prop.minor*10;

		if (cudaCapability >= 200)	{
			uint16_t sumOfCapacity = 0;
			for (uint8_t i = 0; i < numberOfResources; ++i)
				sumOfCapacity += capacityOfResources[i];

			cudaData.sumOfCapacities = sumOfCapacity;
			cudaData.maximalCapacityOfResource = *max_element(capacityOfResources, capacityOfResources+numberOfResources);

			numberOfThreadsPerBlock = 512;
		}	else	{
			numberOfThreadsPerBlock = 128;
		}

		// Merge results array.
		dynSharedMemSize += numberOfThreadsPerBlock*sizeof(MoveInfo);	// merge array

		if ((numberOfActivities-2)*SWAP_RANGE < USHRT_MAX)
			dynSharedMemSize += numberOfThreadsPerBlock*sizeof(uint16_t); // merge help array
		else
			dynSharedMemSize += numberOfThreadsPerBlock*sizeof(uint32_t); // merge help array

		dynSharedMemSize += numberOfActivities*sizeof(uint16_t);	// block order
		dynSharedMemSize += (numberOfResources+1)*sizeof(uint16_t);	// resources indices
		dynSharedMemSize += numberOfActivities*sizeof(uint8_t);		// duration of activities
	} else {
		cudaError = errorHandler(-1);
	}

	/* COPY SUCCESSORS TO CUDA TEXTURE MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaSuccessorsArray, numOfElSuc*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(-1);
	}
	if (!cudaError && cudaMemcpy(cudaSuccessorsArray, linSucArray, numOfElSuc*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(0);
	}
	if (!cudaError && bindTexture(cudaSuccessorsArray, numOfElSuc, 1) != cudaSuccess)	{
		cudaError = errorHandler(0);
	}
	/* COPY PREDECESSORS TO CUDA TEXTURE MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaPredecessorsArray, numOfElPred*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(1);
	}
	if (!cudaError && cudaMemcpy(cudaPredecessorsArray, linPredArray, numOfElPred*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(2);
	}
	if (!cudaError && bindTexture(cudaPredecessorsArray, numOfElPred, 3) != cudaSuccess)	{
		cudaError = errorHandler(2);
	}
	/* COPY ACTIVITIES DURATION TO CUDA */
	if (!cudaError && cudaMalloc((void**) &cudaData.activitiesDuration, numberOfActivities*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(3);
	}
	if (!cudaError && cudaMemcpy(cudaData.activitiesDuration, activitiesDuration, sizeof(uint8_t)*numberOfActivities, cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(4);
	}

	/* COPY SUCCESSORS AND PREDECESSORS INDICES TO TEXTURE MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaSuccessorsIdxsArray, sizeof(uint16_t)*(numberOfActivities+1)) != cudaSuccess)	{
		cudaError = errorHandler(4);
	}
	if (!cudaError && cudaMemcpy(cudaSuccessorsIdxsArray, sucIdxs, sizeof(uint16_t)*(numberOfActivities+1), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(5);
	}
	if (!cudaError && bindTexture(cudaSuccessorsIdxsArray, numberOfActivities+1, 2) != cudaSuccess)	{
		cudaError = errorHandler(5);
	}
	if (!cudaError && cudaMalloc((void**) &cudaPredecessorsIdxsArray, sizeof(uint16_t)*(numberOfActivities+1)) != cudaSuccess)	{
		cudaError = errorHandler(6);
	}
	if (!cudaError && cudaMemcpy(cudaPredecessorsIdxsArray, predIdxs, sizeof(uint16_t)*(numberOfActivities+1), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(7);
	}
	if (!cudaError && bindTexture(cudaPredecessorsIdxsArray, numberOfActivities+1, 4) != cudaSuccess)	{
		cudaError = errorHandler(7);
	}

	/* CONVERT ACTIVITIES RESOURCE REQUIREMENTS TO 1D ARRAY */
	uint32_t resourceReqSize = numberOfActivities*numberOfResources;
	uint8_t *reqResLin = new uint8_t[resourceReqSize], *resWr = reqResLin;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		for (uint8_t r = 0; r < numberOfResources; ++r)	{
			*(resWr++) = activitiesResources[i][r];
		}
	}

	/* COPY ACTIVITIES RESOURCE REQUIREMENTS TO TEXTURE MEMORY */

	if (!cudaError && cudaMalloc((void**) &cudaActivitiesResourcesArray, sizeof(uint8_t)*resourceReqSize) != cudaSuccess)	{
		cudaError = errorHandler(8);
	}
	if (!cudaError && cudaMemcpy(cudaActivitiesResourcesArray, reqResLin, sizeof(uint8_t)*resourceReqSize, cudaMemcpyHostToDevice)	!= cudaSuccess)	{
		cudaError = errorHandler(9);
	}
	if (!cudaError && bindTexture(cudaActivitiesResourcesArray, resourceReqSize, 0) != cudaSuccess)	{
		cudaError = errorHandler(9);
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
	
	// If is possible to run 2 block (shared memory restriction) then copy successorsMatrix to shared memory.
	if (dynSharedMemSize+successorsMatrixSize*sizeof(uint8_t) < 8000)	{
		cout<<"Successors matrix will be copied to shared memory."<<endl;
		dynSharedMemSize += successorsMatrixSize*sizeof(uint8_t);
		cudaData.copySuccessorsMatrixToSharedMemory = true;
	} else	{
		cudaData.copySuccessorsMatrixToSharedMemory = false;
	}
	cudaData.successorsMatrixSize = successorsMatrixSize;

	/* CREATE INITIAL START SET SOLUTIONS */
	srand(time(NULL));
	numberOfBlock = numberOfMultiprocessor*NUMBER_OF_BLOCK_PER_MULTIPROCESSOR;
	SolutionInfo *infoAboutSchedules = new SolutionInfo[SOLUTION_SET_SIZE];
	uint16_t *randomSchedules = new uint16_t[SOLUTION_SET_SIZE*numberOfActivities], *schedWr = randomSchedules;

	// Rewrite to sets.
	for (uint16_t b = 0; b < SOLUTION_SET_SIZE; ++b)	{
		makeDiversification(activitiesOrder, successorsMatrix, 100);
/*
		cout<<"Cost before: "<<evaluateOrder(activitiesOrder)<<endl;
		for (int32_t i = 1; i < numberOfActivities-2; ++i)	{ 
			uint32_t bestI = i, bestJ = i;
			uint32_t bestCost = evaluateOrder(activitiesOrder);
			for (int32_t j = i+1; j < numberOfActivities-1; ++j)   {
				if (checkSwapPrecedencePenalty(activitiesOrder, successorsMatrix, i,j))	{
					swap(activitiesOrder[i], activitiesOrder[j]);
					uint32_t cost = evaluateOrder(activitiesOrder);
					if (cost < bestCost)	{
						bestI = i; bestJ = j;
						bestCost = cost;
					}
					swap(activitiesOrder[i], activitiesOrder[j]);
				}
			}
			if (bestI != bestJ)
				swap(activitiesOrder[bestI], activitiesOrder[bestJ]);
		}
		uint16_t * startById = new uint16_t[numberOfActivities];
		cout<<"Cost after: "<<evaluateOrder(activitiesOrder, NULL, startById)<<endl;
		cout<<"Penalty after: "<<computePrecedencePenalty(startById)<<endl;
		delete[] startById; */
		
		infoAboutSchedules[b].readCounter = 0;
		infoAboutSchedules[b].solutionCost = evaluateOrder(activitiesOrder);
		schedWr = copy(activitiesOrder, activitiesOrder+numberOfActivities, schedWr);
	}

	/* COPY INITIAL SOLUTIONS TO CUDA MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaData.solutionsSetInfo, sizeof(SolutionInfo)*cudaData.solutionsSetSize) != cudaSuccess)	{
		cudaError = errorHandler(10);
	}
	if (!cudaError && cudaMemcpy(cudaData.solutionsSetInfo, infoAboutSchedules, cudaData.solutionsSetSize*sizeof(SolutionInfo), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(11);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.solutionsSet, sizeof(uint16_t)*cudaData.solutionsSetSize*numberOfActivities) != cudaSuccess)	{
		cudaError = errorHandler(11);
	}
	if (!cudaError && cudaMemcpy(cudaData.solutionsSet, randomSchedules, sizeof(int16_t)*cudaData.solutionsSetSize*numberOfActivities, cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(12);
	}

	uint32_t tabuListsSize = numberOfBlock*cudaData.maxTabuListSize;
	uint32_t tabuCachesSize = numberOfActivities*numberOfActivities*numberOfBlock;

	/* COPY TABU LISTS TO CUDA MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaData.tabuLists, tabuListsSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(12);
	}
	if (!cudaError && cudaMemset(cudaData.tabuLists, 0, tabuListsSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(13);
	}
	/* CREATE TABU CACHE */
	if (!cudaError && cudaMalloc((void**) &cudaData.tabuCaches, tabuCachesSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(13);
	}
	if (!cudaError && cudaMemset(cudaData.tabuCaches, 0, tabuCachesSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(14);
	}

	/* CONVERT CAPACITIES OF RESOURCES TO 1D ARRAY */
	uint16_t *resIdxs = new uint16_t[numberOfResources+1];
	resIdxs[0] = 0;
	for (uint8_t r = 0; r < numberOfResources; ++r)	{
		resIdxs[r+1] =  resIdxs[r]+capacityOfResources[r];
	}

	/* COPY RESOURCES CAPACITIES TO CUDA MEMORY */
	if (!cudaError && cudaMalloc((void**) &cudaData.resourceIndices, (numberOfResources+1)*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(14);
	}
	if (!cudaError && cudaMemcpy(cudaData.resourceIndices, resIdxs, (numberOfResources+1)*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(15);
	}
	/* CREATE START TIMES ARRAYS */
/*	if (!cudaError && cudaMalloc((void**) &cudaData.startTimesById, numberOfBlock*numberOfThreadsPerBlock*numberOfActivities*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(15);
	} */
	if (!cudaError && cudaMalloc((void**) &cudaData.setStateOfCommunication, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(16);
	}
	/* SET START STATE OF COMMUNICATION */
	if (!cudaError && cudaMemset(cudaData.setStateOfCommunication, DATA_AVAILABLE, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(17);
	}
	/* BEST CURRENT SOLUTIONS OF THE BLOCKS */
	if (!cudaError && cudaMalloc((void**) &cudaData.blocksBestSolution, numberOfActivities*numberOfBlock*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(18);
	}
	/* ALLOCATE AND INIT HASH MAP */
	if (!cudaError && cudaMalloc((void**) &cudaData.hashMap, HASH_TABLE_SIZE*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(19);
	}
	if (!cudaError && cudaMemset(cudaData.hashMap, 0, HASH_TABLE_SIZE*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(20);
	}
	/* COPY SUCCESSORS BIT MATRIX */
	if (!cudaError && cudaMalloc((void**) &cudaData.successorsMatrix, successorsMatrixSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(20);
	}
	if (!cudaError && cudaMemcpy(cudaData.successorsMatrix, successorsMatrix, successorsMatrixSize*sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(21);
	}
	/* CREATE SWAP PENALTY FREE MERGE ARRAYS */
	if (!cudaError && cudaMalloc((void**) &cudaData.swapFreeMergeArray, (numberOfActivities-2)*cudaData.swapRange*sizeof(MoveIndices)*numberOfBlock) != cudaSuccess)	{
		cudaError = errorHandler(21);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.mergeHelpArray, (numberOfActivities-2)*cudaData.swapRange*sizeof(MoveIndices)*numberOfBlock) != cudaSuccess)	{
		cudaError = errorHandler(22);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.solutionSetTabuLists, cudaData.solutionsSetSize*cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(23);
	}
	if (!cudaError && cudaMemset(cudaData.solutionSetTabuLists, 0, cudaData.solutionsSetSize*cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(24);
	}
	/* GLOBAL BEST SOLUTION OF ALL BLOCKS */
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolution, cudaData.numberOfActivities*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(24);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolutionCost, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(25);
	}
	if (!cudaError && cudaMemset(cudaData.globalBestSolutionCost, UCHAR_MAX, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(26);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.globalBestSolutionTabuList, cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(26);
	}
	if (!cudaError && cudaMemset(cudaData.globalBestSolutionTabuList, 0, cudaData.maxTabuListSize*sizeof(MoveIndices)) != cudaSuccess)	{
		cudaError = errorHandler(27);
	}
	if (!cudaError && cudaMalloc((void**) &cudaData.globalStateOfCommunication, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(27);
	}
	if (!cudaError && cudaMemset(cudaData.globalStateOfCommunication, DATA_AVAILABLE, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(28);
	}

	for (uint32_t i = 0; i < numberOfActivities; ++i)
		delete[] distanceMatrix[i];
	delete[] distanceMatrix;
	delete[] successorsMatrix;
	delete[] resIdxs;
	delete[] randomSchedules;
	delete[] infoAboutSchedules;
	delete[] reqResLin;
	delete[] linSucArray;
	delete[] linPredArray;
	delete[] predIdxs;
	delete[] sucIdxs;
	delete[] activitiesOrder;

	return cudaError;
}

bool ScheduleSolver::errorHandler(int16_t phase)	{
	cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
	switch (phase)	{
		case 28:
			cudaFree(cudaData.globalStateOfCommunication);
		case 27:
			cudaFree(cudaData.globalBestSolutionTabuList);
		case 26:
			cudaFree(cudaData.globalBestSolutionCost);
		case 25:
			cudaFree(cudaData.globalBestSolution);
		case 24:
			cudaFree(cudaData.solutionSetTabuLists);
		case 23:
			cudaFree(cudaData.mergeHelpArray);
		case 22:
			cudaFree(cudaData.swapFreeMergeArray);
		case 21:
			cudaFree(cudaData.successorsMatrix);
		case 20:
			cudaFree(cudaData.hashMap);
		case 19:
			cudaFree(cudaData.blocksBestSolution);
		case 18:
		case 17:
			cudaFree(cudaData.setStateOfCommunication);
		case 16:
	//		cudaFree(cudaData.startTimesById);
		case 15:
			cudaFree(cudaData.resourceIndices);
		case 14:
			cudaFree(cudaData.tabuCaches);
		case 13:
			cudaFree(cudaData.tabuLists);
		case 12:
			cudaFree(cudaData.solutionsSet);
		case 11:
			cudaFree(cudaData.solutionsSetInfo);
		case 10:
			unbindTexture(0);	// activities resources requirements
		case 9:
			cudaFree(cudaActivitiesResourcesArray);
		case 8:
			unbindTexture(4);	// predecessors indices
		case 7:
			cudaFree(cudaPredecessorsIdxsArray);
		case 6:
			unbindTexture(2);	// successors indices
		case 5:
			cudaFree(cudaSuccessorsIdxsArray);
		case 4:
			cudaFree(cudaData.activitiesDuration);
		case 3:
			unbindTexture(3);	// predecessors
		case 2:
			cudaFree(cudaPredecessorsArray);
		case 1:
			unbindTexture(1);	// successors
		case 0:
			cudaFree(cudaSuccessorsArray);

		default:
			break;
	}
	return true;
}

bool cmpTmp(const SolutionInfo& x, const SolutionInfo& y)	{
	if (x.solutionCost < y.solutionCost)
		return true;
	else
		return false;
}

#include <map> // !!
void ScheduleSolver::solveSchedule(const uint32_t& maxIter, const uint32_t& maxIterToDiversification)	{
	#ifdef __GNUC__
	timeval startTime, endTime, diffTime;
	gettimeofday(&startTime, NULL);
	#elif defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER startTimeStamp, stopTimeStamp;
	QueryPerformanceFrequency(&ticksPerSecond);
	QueryPerformanceCounter(&startTimeStamp);
	#endif

	/* START TO SEARCH BEST SOLUTION */

	cout<<"sharedMemSize: "<<dynSharedMemSize<<endl;
	cout<<"Number of threads: "<<numberOfThreadsPerBlock<<endl;

	cudaData.maximalValueOfReadCounter = 3;
	cudaData.numberOfDiversificationSwaps = 20;
	cudaData.numberOfIterationsPerBlock = maxIter;
	cudaData.maximalIterationsSinceBest = maxIterToDiversification;
	
	runCudaSolveRCPSP(numberOfBlock, numberOfThreadsPerBlock, cudaCapability, dynSharedMemSize, cudaData);

	/* FIND BEST BLOCK SOLUTION */

	cout<<"sizeof(MoveIndices): "<<sizeof(MoveIndices)<<endl;
	// !!!!
	bool cudaError = false;
	uint32_t bestGlobalCost;
	if (!cudaError && cudaMemcpy(&bestGlobalCost, cudaData.globalBestSolutionCost, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError && cudaMemcpy(bestScheduleOrder, cudaData.globalBestSolution, sizeof(uint16_t)*numberOfActivities, cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError)	{
		if (bestGlobalCost < 0xffffffff)	{
			solutionComputed = true;
		}
		cout<<"Cuda best cost: "<<bestGlobalCost<<endl;
	}

	if (cudaError)	{
		throw runtime_error("ScheduleSolver::solveSchedule: Error occur when try to solve the instance!");
	}

/*
	uint32_t *hashMap = new uint32_t[HASH_TABLE_SIZE];
	cudaMemcpy(hashMap, cudaData.hashMap, sizeof(uint32_t)*HASH_TABLE_SIZE, cudaMemcpyDeviceToHost);
	
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
	cout<<"Total accuracy: "<<numberCorrect/((double) numberCollision+numberCorrect)<<endl;

	delete[] hashMap; */

	#ifdef __GNUC__
	gettimeofday(&endTime, NULL);
	timersub(&endTime, &startTime, &diffTime);
	totalRunTime = diffTime.tv_sec+diffTime.tv_usec/1000000.;
	#elif defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
	QueryPerformanceCounter(&stopTimeStamp);
	totalRunTime = (stopTimeStamp.QuadPart-startTimeStamp.QuadPart)/((double) ticksPerSecond.QuadPart);
	#endif
}

uint16_t ScheduleSolver::evaluateOrder(const uint16_t *order, uint16_t *startTimesWriter, uint16_t *startTimesWriterById)	const	{
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

uint32_t ScheduleSolver::computePrecedencePenalty(const uint16_t *startTimesById)	const	{
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

void ScheduleSolver::printSchedule(uint16_t *scheduleOrder, bool verbose, ostream& OUT)	const	{
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

void ScheduleSolver::makeDiversification(uint16_t * const& order, const uint8_t * const& successorsMatrix, const uint32_t numberOfSwaps)	{
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
	for (int i = 0; i < 5; ++i)
		unbindTexture(i);
	cudaFree(cudaData.mergeHelpArray);
	cudaFree(cudaData.activitiesDuration);
	cudaFree(cudaSuccessorsArray);
	cudaFree(cudaPredecessorsArray);
	cudaFree(cudaSuccessorsIdxsArray);
	cudaFree(cudaPredecessorsIdxsArray);
	cudaFree(cudaActivitiesResourcesArray);
	cudaFree(cudaData.successorsMatrix);
	cudaFree(cudaData.solutionsSetInfo);
	cudaFree(cudaData.solutionsSet);
	cudaFree(cudaData.tabuLists);
	cudaFree(cudaData.tabuCaches);
	cudaFree(cudaData.resourceIndices);
//	cudaFree(cudaData.startTimesById);
	cudaFree(cudaData.setStateOfCommunication);
	cudaFree(cudaData.blocksBestSolution);
	cudaFree(cudaData.hashMap);
	cudaFree(cudaData.swapFreeMergeArray);
	cudaFree(cudaData.solutionSetTabuLists);
	cudaFree(cudaData.globalBestSolution);
	cudaFree(cudaData.globalBestSolutionCost);
	cudaFree(cudaData.globalBestSolutionTabuList);
	cudaFree(cudaData.globalStateOfCommunication);
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



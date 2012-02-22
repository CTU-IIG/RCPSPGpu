#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <stdexcept>

#include <map> // !!!

#ifdef __GNUC__
#include <sys/time.h>
#elif defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
#include <Windows.h>
#endif

#define USE_SHARED_LOAD 1

#include "ConfigureRCPSP.h"
#include "CudaConstants.h"
#include "ScheduleSolver.cuh"
#include "SourcesLoad.h"

__constant__ uint8_t cudaActivitiesDuration[NUMBER_OF_ACTIVITIES];
__constant__ uint8_t cudaActivitiesResources[NUMBER_OF_ACTIVITIES*NUMBER_OF_RESOURCES];

texture<uint16_t,1,cudaReadModeElementType> cudaSuccessorsTex;
texture<uint16_t,1,cudaReadModeElementType> cudaPredecessorsTex;

/* Blocks communication. */
#define NOT_WRITED 0
#define WRITING_DATA 1
#define READING_DATA 2
#define DATA_AVAILABLE 3

/* HASH CONSTANTS */
// #define R 1779033703, 3144134277, 1013904242, 2773480762, 1359893119, 2600822924, 528734635, 1541459225 
//				0.95125,	0.9625,			-	,	-		,	0.89375, 		-  ,   0.94625,  0.94
#define R 3144134277
#define HASH_TABLE_SIZE 16777216

#include "CudaFunctions.cu"

using namespace std;

ScheduleSolver::ScheduleSolver(uint8_t resNum, uint8_t *capRes, uint16_t actNum, uint8_t *actDur, uint16_t **actSuc, uint16_t *actNumSuc, uint8_t **actRes, bool verbose)
		: numberOfResources(resNum), capacityOfResources(capRes), numberOfActivities(actNum), activitiesDuration(actDur),
		  activitiesSuccessors(actSuc), numberOfSuccessors(actNumSuc), activitesResources(actRes), totalRunTime(-1)	{

	uint16_t *activitiesOrder = new uint16_t[numberOfActivities];
	uint16_t *levelsCounter = createInitialSolution(activitiesOrder);
	if (prepareCudaMemory(activitiesOrder, levelsCounter, verbose) == true)	{
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

uint16_t* ScheduleSolver::createInitialSolution(uint16_t *activitiesOrder)	{

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
	uint16_t *levelsCounter = new uint16_t[deep+1];
	memset(levelsCounter, 0, sizeof(uint16_t)*(deep+1));
	for (uint16_t curDeep = 0; curDeep < deep; ++curDeep)	{
		for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
			if (levels[activityId] == curDeep)	{
				activitiesOrder[schedIdx++] = activityId;
				++levelsCounter[curDeep+1];
			}
		}
	}
	levelsCounter[0] = deep;

	delete[] currentLevel;
	delete[] newCurrentLevel;

	return levelsCounter;
}

bool ScheduleSolver::prepareCudaMemory(uint16_t *activitiesOrder, uint16_t *levelsCounter, bool verbose)	{

	bool cudaError = false;
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

		if (cudaCapability > 200)	{
			uint16_t sumOfCapacity = 0;
			for (uint8_t i = 0; i < numberOfResources; ++i)
				sumOfCapacity += capacityOfResources[i];

			if (sumOfCapacity > 100)
				numberOfThreadsPerBlock = 512;
			else
				numberOfThreadsPerBlock = 256;

			cudaFuncSetCacheConfig(solveRCPSP, cudaFuncCachePreferL1);
		}	else	{
			#if USE_SHARED_LOAD == 1
			numberOfThreadsPerBlock = 32;
			dynSharedMemSize += (TOTAL_SUM_OF_CAPACITY*2+MAXIMUM_CAPACITY_OF_RESOURCE*3)*numberOfThreadsPerBlock;	
			#define SHARED_ALLOC 1
			#else
			numberOfThreadsPerBlock = 128;
			#endif
		}
		dynSharedMemSize += numberOfThreadsPerBlock*sizeof(uint64_t);
	} else {
		cudaError = errorHandler(-1);
	}

	if (!cudaError && cudaMalloc((void**) &cudaSuccessorsArray, numOfElSuc*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(-1);
	}
	if (!cudaError && cudaMemcpy(cudaSuccessorsArray, linSucArray, numOfElSuc*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(0);
	}
	if (!cudaError && cudaBindTexture(NULL, cudaSuccessorsTex, cudaSuccessorsArray, numOfElSuc*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(0);
	}
	if (!cudaError && cudaMalloc((void**) &cudaPredecessorsArray, numOfElPred*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(1);
	}
	if (!cudaError && cudaMemcpy(cudaPredecessorsArray, linPredArray, numOfElPred*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(2);
	}
	if (!cudaError && cudaBindTexture(NULL, cudaPredecessorsTex, cudaPredecessorsArray, numOfElPred*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(2);
	}
	if (!cudaError && cudaMemcpyToSymbol(cudaActivitiesDuration, activitiesDuration, numberOfActivities*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(3);
	}
	if (!cudaError && cudaMalloc((void**) &cudaSuccessorsIdxs, sizeof(uint16_t)*(numberOfActivities+1)) != cudaSuccess)	{
		cudaError = errorHandler(3);
	}
	if (!cudaError && cudaMemcpy(cudaSuccessorsIdxs, sucIdxs, sizeof(uint16_t)*(numberOfActivities+1), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(4);
	}
	if (!cudaError && cudaMalloc((void**) &cudaPredecessorsIdxs, sizeof(uint16_t)*(numberOfActivities+1)) != cudaSuccess)	{
		cudaError = errorHandler(4);
	}
	if (!cudaError && cudaMemcpy(cudaPredecessorsIdxs, predIdxs, sizeof(uint16_t)*(numberOfActivities+1), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(5);
	}

	uint32_t resourceReqSize = numberOfActivities*numberOfResources;
	uint8_t *reqResLin = new uint8_t[resourceReqSize], *resWr = reqResLin;
	for (uint16_t i = 0; i < numberOfActivities; ++i)	{
		for (uint8_t r = 0; r < numberOfResources; ++r)	{
			*(resWr++) = activitesResources[i][r];
		}
	}

	if (!cudaError && cudaMemcpyToSymbol(cudaActivitiesResources, reqResLin, resourceReqSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(5);
	}

	srand(time(NULL));
	uint16_t deep = levelsCounter[0];
	numberOfBlock = numberOfMultiprocessor*NUMBER_OF_BLOCK_PER_MULTIPROCESSOR;
	uint32_t *costOfSchedules = new uint32_t[numberOfBlock];
	uint16_t *randomSchedules = new uint16_t[numberOfBlock*numberOfActivities], *schedWr = randomSchedules;
/*	cout<<"deep: "<<deep<<endl;
	for (int i = 0; i < deep; ++i)
		cout<<" "<<levelsCounter[i+1];
	cout<<endl; */

	++levelsCounter;
	for (uint16_t i = 0; i < numberOfBlock; ++i)	{
		costOfSchedules[i] = evaluateOrder(activitiesOrder);
		schedWr = copy(activitiesOrder, activitiesOrder+numberOfActivities, schedWr);
		
		uint16_t startIdx = 0;
		for (uint16_t i = 0; i < deep; ++i)	{
			uint16_t endIdx = startIdx+levelsCounter[i];
			random_shuffle(activitiesOrder+startIdx, activitiesOrder+endIdx);
			startIdx = endIdx;
		}
	}
	--levelsCounter;

	if (!cudaError && cudaMalloc((void**) &cudaBestBlocksCost, sizeof(uint32_t)*numberOfBlock) != cudaSuccess)	{
		cudaError = errorHandler(5);
	}
	if (!cudaError && cudaMemcpy(cudaBestBlocksCost, costOfSchedules, numberOfBlock*sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(6);
	}
	if (!cudaError && cudaMalloc((void**) &cudaBestBlocksOrder, sizeof(uint16_t)*numberOfBlock*numberOfActivities) != cudaSuccess)	{
		cudaError = errorHandler(6);
	}
	if (!cudaError && cudaMemcpy(cudaBestBlocksOrder, randomSchedules, sizeof(int16_t)*numberOfBlock*numberOfActivities, cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(7);
	}

	uint32_t tabuListsSize = numberOfBlock*TABU_LIST_SIZE;
	uint32_t tabuCachesSize = numberOfActivities*numberOfActivities*numberOfBlock;

	if (!cudaError && cudaMalloc((void**) &cudaTabuLists, tabuListsSize*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(7);
	}
	if (!cudaError && cudaMemset(cudaTabuLists, UCHAR_MAX, tabuListsSize*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(8);
	}
	if (!cudaError && cudaMalloc((void**) &cudaTabuCaches, tabuCachesSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(8);
	}
	if (!cudaError && cudaMemset(cudaTabuCaches, 0, tabuCachesSize*sizeof(uint8_t)) != cudaSuccess)	{
		cudaError = errorHandler(9);
	}
	if (!cudaError && cudaMalloc((void**) &cudaCapacityIdxs, (numberOfResources+1)*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(9);
	}

	uint16_t *resIdxs = new uint16_t[numberOfResources+1];
	resIdxs[0] = 0;
	for (uint8_t r = 0; r < numberOfResources; ++r)	{
		resIdxs[r+1] =  resIdxs[r]+capacityOfResources[r];
	}

	if (!cudaError && cudaMemcpy(cudaCapacityIdxs, resIdxs, (numberOfResources+1)*sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess)	{
		cudaError = errorHandler(10);
	}
	if (!cudaError && cudaMalloc((void**) &cudaStartTimesById, numberOfBlock*numberOfThreadsPerBlock*numberOfActivities*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(10);
	}
	if (!cudaError && cudaMalloc((void**) &cudaStateOfCommunication, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(11);
	}
	if (!cudaError && cudaMemset(cudaStateOfCommunication, NOT_WRITED, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(12);
	}
	if (!cudaError && cudaMalloc((void**) &cudaBlocksBestEval, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(12);
	}
	if (!cudaError && cudaMemset(cudaBlocksBestEval, UCHAR_MAX, sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(13);
	}
	if (!cudaError && cudaMalloc((void**) &cudaBlocksBestSolution, numberOfActivities*sizeof(uint16_t)) != cudaSuccess)	{
		cudaError = errorHandler(13);
	}
	if (!cudaError && cudaMalloc((void**) &cudaHashMap, HASH_TABLE_SIZE*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(14);
	}
	if (!cudaError && cudaMemset(cudaHashMap, 0, HASH_TABLE_SIZE*sizeof(uint32_t)) != cudaSuccess)	{
		cudaError = errorHandler(15);
	}

	delete[] resIdxs;
	delete[] randomSchedules;
	delete[] costOfSchedules;
	delete[] reqResLin;
	delete[] linSucArray;
	delete[] linPredArray;
	delete[] predIdxs;
	delete[] sucIdxs;
	delete[] activitiesOrder;
	delete[] levelsCounter;

	return cudaError;
}

bool ScheduleSolver::errorHandler(int16_t phase)	{
	cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
	switch (phase)	{
		case 15:
			cudaFree(cudaHashMap);
		case 14:
			cudaFree(cudaBlocksBestSolution);
		case 13:
			cudaFree(cudaBlocksBestEval);
		case 12:
			cudaFree(cudaStateOfCommunication);
		case 11:
			cudaFree(cudaStartTimesById);
		case 10:
			cudaFree(cudaCapacityIdxs);
		case 9:
			cudaFree(cudaTabuCaches);
		case 8:
			cudaFree(cudaTabuLists);
		case 7:
			cudaFree(cudaBestBlocksOrder);
		case 6:
			cudaFree(cudaBestBlocksCost);
		case 5:
			cudaFree(cudaPredecessorsIdxs);
		case 4:
			cudaFree(cudaSuccessorsIdxs);
		case 3:
			cudaUnbindTexture(cudaPredecessorsTex);
		case 2:
			cudaFree(cudaPredecessorsArray);
		case 1:
			cudaUnbindTexture(cudaSuccessorsTex);	
		case 0:
			cudaFree(cudaSuccessorsArray);

		default:
			break;
	}
	return true;
}

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
/*
	cout<<"sharedMemSize: "<<dynSharedMemSize<<endl;
	cout<<"Number of threads: "<<numberOfThreadsPerBlock<<endl;
	*/

//	numberOfBlock = 1;
	solveRCPSP<<<numberOfBlock,numberOfThreadsPerBlock,dynSharedMemSize>>>(numberOfActivities, numberOfResources, cudaSuccessorsIdxs, cudaPredecessorsIdxs,
			cudaBestBlocksCost, cudaBestBlocksOrder, cudaTabuLists, cudaTabuCaches, cudaCapacityIdxs, cudaStartTimesById, maxIter, maxIterToDiversification,
			cudaStateOfCommunication, cudaBlocksBestEval, cudaBlocksBestSolution, cudaHashMap);

	/* FIND BEST BLOCK SOLUTION */

	bool cudaError = false;
	uint32_t *bestBlockResults = new uint32_t[numberOfBlock];
	uint16_t *bestBlockSchedules = new uint16_t[numberOfBlock*numberOfActivities];
	if (!cudaError && cudaMemcpy(bestBlockSchedules, cudaBestBlocksOrder, sizeof(uint16_t)*numberOfBlock*numberOfActivities, cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError && cudaMemcpy(bestBlockResults, cudaBestBlocksCost, sizeof(uint32_t)*numberOfBlock, cudaMemcpyDeviceToHost) != cudaSuccess)	{
		cerr<<"Cuda error: "<<cudaGetErrorString(cudaGetLastError())<<endl;
		cudaError = true;	
	}
	if (!cudaError)	{
		uint32_t *bestCost = min_element(bestBlockResults, bestBlockResults+numberOfBlock);
		uint16_t *bestSchedulePtr = (bestCost-bestBlockResults)*numberOfActivities+bestBlockSchedules, *ptrWr = bestScheduleOrder;
		copy(bestSchedulePtr, bestSchedulePtr+numberOfActivities, ptrWr);
/*
		uint32_t i = 0;
		for (uint16_t *orders = bestBlockSchedules; orders < bestBlockSchedules+numberOfActivities*numberOfBlock; orders += numberOfActivities)	{
			printSchedule(orders);
			cout<<"Writed best cost: "<<bestBlockResults[i++]<<endl;
		}
		*/
	}

	delete[] bestBlockSchedules;
	delete[] bestBlockResults;

	if (cudaError)	{
		throw runtime_error("ScheduleSolver::solveSchedule: Error occur when try to solve the instance!");
	}

	uint32_t *hashMap = new uint32_t[HASH_TABLE_SIZE];
	cudaMemcpy(hashMap, cudaHashMap, sizeof(uint32_t)*HASH_TABLE_SIZE, cudaMemcpyDeviceToHost);
	
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

	delete[] hashMap;

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

		start = max(load.getEarliestStartTime(activitesResources[activityId]), start);
		load.addActivity(start, start+activitiesDuration[activityId], activitesResources[activityId]);
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
	return PRECEDENCE_PENALTY*penalty;
}

void ScheduleSolver::printSchedule(uint16_t *scheduleOrder, bool verbose, ostream& OUT)	const	{
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
		OUT<<"Schedule solve time: "<<totalRunTime<<" s"<<endl;
	} else {
		OUT<<scheduleLength<<"+"<<precedencePenalty<<"\t["<<totalRunTime<<" s]"<<endl;
	}

	delete[] startTimesById;
	delete[] startTimes;
}

void ScheduleSolver::printBestSchedule(bool verbose, ostream& OUT)	const	{
	printSchedule(bestScheduleOrder, verbose, OUT);
}

void ScheduleSolver::freeCudaMemory()	{
	cudaUnbindTexture(cudaSuccessorsTex);
	cudaUnbindTexture(cudaPredecessorsTex);
	cudaFree(cudaSuccessorsArray);
	cudaFree(cudaPredecessorsArray);
	cudaFree(cudaSuccessorsIdxs);
	cudaFree(cudaPredecessorsIdxs);
	cudaFree(cudaBestBlocksCost);
	cudaFree(cudaBestBlocksOrder);
	cudaFree(cudaTabuLists);
	cudaFree(cudaTabuCaches);
	cudaFree(cudaCapacityIdxs);
	cudaFree(cudaStartTimesById);
	cudaFree(cudaStateOfCommunication);
	cudaFree(cudaBlocksBestEval);
	cudaFree(cudaBlocksBestSolution);
	cudaFree(cudaHashMap);
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



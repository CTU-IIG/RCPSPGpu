/*!
 * \file CudaFunctions.cu
 * \author Libor Bukata
 * \brief RCPSP Cuda functions.
 */

#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include "CudaConstants.h"
#include "CudaFunctions.cuh"

#if defined _WIN32 || defined _WIN64 || defined WIN32 || defined WIN64
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

using std::cerr;
using std::cout;
using std::endl;

//! Texture reference of activities resource requirements.
texture<uint8_t,1,cudaReadModeElementType> cudaActivitiesResourcesTex;
//! Texture reference of predecessors.
texture<uint16_t,1,cudaReadModeElementType> cudaPredecessorsTex;
//! Texture reference of predecessors indices.
texture<uint16_t,1,cudaReadModeElementType> cudaPredecessorsIndicesTex;
//! Texture reference of successors.
texture<uint16_t,1,cudaReadModeElementType> cudaSuccessorsTex;
//! Texture reference of successors indices.
texture<uint16_t,1,cudaReadModeElementType> cudaSuccessorsIndicesTex;

//! The longest paths from the end dummy activity to the others in the transformed graph.
__constant__ uint16_t rightLeftLongestPaths[NUMBER_OF_ACTIVITIES];


/* CUDA BIND TEXTURES */

int bindTexture(void *texData, int32_t arrayLength, int option)	{
	switch (option)	{
		case ACTIVITIES_RESOURCES:
			return cudaBindTexture(NULL, cudaActivitiesResourcesTex, texData, arrayLength*sizeof(uint8_t));
		case PREDECESSORS:
			return cudaBindTexture(NULL, cudaPredecessorsTex, texData, arrayLength*sizeof(uint16_t));
		case PREDECESSORS_INDICES:
			return cudaBindTexture(NULL, cudaPredecessorsIndicesTex, texData, arrayLength*sizeof(uint16_t));
		case SUCCESSORS:
			return cudaBindTexture(NULL, cudaSuccessorsTex, texData, arrayLength*sizeof(uint16_t));
		case SUCCESSORS_INDICES:
			return cudaBindTexture(NULL, cudaSuccessorsIndicesTex, texData, arrayLength*sizeof(uint16_t));
		default:
			cerr<<"bindTextures: Invalid option!"<<endl;
	}
	return cudaErrorInvalidValue;
}

int unbindTexture(int option)	{
	switch (option)	{
		case ACTIVITIES_RESOURCES:
			return cudaUnbindTexture(cudaActivitiesResourcesTex);
		case PREDECESSORS:
			return cudaUnbindTexture(cudaPredecessorsTex);
		case PREDECESSORS_INDICES:
			return cudaUnbindTexture(cudaPredecessorsIndicesTex);
		case SUCCESSORS:
			return cudaUnbindTexture(cudaSuccessorsTex);
		case SUCCESSORS_INDICES:
			return cudaUnbindTexture(cudaSuccessorsIndicesTex);
		default:
			cerr<<"unbindTextures: Invalid option!"<<endl;
	}
	return cudaErrorInvalidValue;
}

int memcpyToSymbol(void *source, int32_t arrayLength, int option)	{
	switch (option)	{
		case THE_LONGEST_PATHS:
			return cudaMemcpyToSymbol(rightLeftLongestPaths, (void*) source, arrayLength*sizeof(uint16_t));
		default:
			cerr<<"memcpyToSymbol: Invalid option!"<<endl;
	}
	return cudaErrorInvalidValue;
}


/* CUDA IMPLEMENT OF SOURCES LOAD - CAPACITY RESOLUTION  */

/*!
 * \param cudaData RCPSP constants, variables, ...
 * \param resourcesLoad Array of the earliest resource start times.
 * \param startValues Helper array for resource evaluation.
 * \brief Prepare arrays for next use (schedule evaluation).
 */
inline __device__ void cudaPrepareArrays(const CudaData& cudaData, uint16_t *& resourcesLoad, uint16_t *& startValues)	{
	for (uint16_t i = 0; i < cudaData.sumOfCapacities; ++i)
		resourcesLoad[i] = 0;
	for (uint16_t i = 0; i < cudaData.maximalCapacityOfResource; ++i)
		startValues[i] = 0;
}

/*!
 * \param numberOfResources Number of resources.
 * \param activityId Activity identification.
 * \param resourcesLoad Array of the earliest resource start times.
 * \param resourceIndices Access indices for resources.
 * \return Earliest start time of an activity.
 * \brief Function return earliest possible start time of an activity. Precedence relations are ignored.
 */
inline __device__ uint16_t cudaGetEarliestStartTime(const uint16_t& numberOfResources, const uint16_t& activityId, uint16_t *&resourcesLoad, uint16_t *&resourceIndices) {
	uint16_t bestStart = 0;
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		uint8_t activityRequirement = tex1Dfetch(cudaActivitiesResourcesTex, activityId*numberOfResources+resourceId);
		if (activityRequirement > 0)
			bestStart = max(resourcesLoad[resourceIndices[resourceId+1]-activityRequirement], bestStart);
	}
	return bestStart;
}

/*!
 * \param activityId Activity identification.
 * \param activityStart Start time of an activity.
 * \param activityStop Stop time of an activity.
 * \param numberOfResources Number of resources.
 * \param resourceIndices Access indices for resources.
 * \param resourcesLoad Array of the earliest resource start times.
 * \param startValues Helper array for resource evaluation.
 * \brief Function add new activity and update resources arrays. Irreversible process.
 */
inline __device__ void cudaAddActivity(const uint16_t& activityId, const uint16_t& activityStart, const uint16_t& activityStop,
		const uint16_t& numberOfResources, uint16_t *&resourceIndices,  uint16_t *&resourcesLoad, uint16_t *&startValues)	{
	
	int32_t requiredSquares, timeDiff;
	int32_t c, k, capacityOfResource, resourceRequirement, newStartTime, resourceStartIdx;
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		resourceStartIdx = resourceIndices[resourceId];
		capacityOfResource = resourceIndices[resourceId+1]-resourceStartIdx;
		resourceRequirement = tex1Dfetch(cudaActivitiesResourcesTex, activityId*numberOfResources+resourceId);
		requiredSquares = resourceRequirement*(activityStop-activityStart);
		if (requiredSquares > 0)	{
			c = 0; k = 0;
			newStartTime = activityStop;
			while (requiredSquares > 0 && k < capacityOfResource)	{
				if (resourcesLoad[resourceStartIdx+k] < newStartTime)	{
					if (c >= resourceRequirement)
						newStartTime = startValues[c-resourceRequirement];
					timeDiff = newStartTime-max(resourcesLoad[resourceStartIdx+k], activityStart);
					if (requiredSquares-timeDiff > 0)	{
						requiredSquares -= timeDiff;
						startValues[c++] = resourcesLoad[resourceStartIdx+k];
						resourcesLoad[resourceStartIdx+k] = newStartTime;
					} else {
						resourcesLoad[resourceStartIdx+k] = newStartTime-timeDiff+requiredSquares;
						break;
					}
				}
				++k;
			}
		}
	}
}

/* CUDA IMPLEMENT OF SOURCES LOAD - TIME RESOLUTION  */

/*!
 * \param numberOfActivities Number of activities in the project.
 * \param numberOfResources Number of renewable resources in the project.
 * \param UBTime Upper bound of the maximal duration of the project.
 * \param remainingResourcesCapacity Free capacity of each resource with respect to time.
 * \param resourceIndices Access indices for resources.
 * \brief It initializes vectors of free capacities to initial values (capacities of resources).
 */
inline __device__ void cudaPrepareArrays(const uint16_t& numberOfActivities, const uint16_t& numberOfResources, const uint32_t& UBTime,
	       	uint8_t *& remainingResourcesCapacity, uint16_t *& resourceIndices)	{
	for (uint16_t resourceId = 0; resourceId < numberOfResources; ++resourceId)
		for (uint32_t t = 0; t < UBTime; ++t)
			remainingResourcesCapacity[resourceId*UBTime+t] = resourceIndices[resourceId+1]-resourceIndices[resourceId];
}

/*!
 * \param numberOfResources Number of renewable resources in the project.
 * \param activityId Identification of the activity that should be added (required for texture memory access).
 * \param remainingResourcesCapacity Free capacity of each resource with respect to time.
 * \param precTime The earliest precedence violation free start time of the activity activityId.
 * \param activityDuration Duration of the activity activityId.
 * \param UBTime Upper bound of the maximal duration of the project.
 * \return The earliest start time of the activity without resource overload.
 * \brief It finds out the earliest start time of the activity activityId.
 */
inline __device__ uint16_t cudaGetEarliestStartTime(const uint16_t& numberOfResources, const uint16_t& activityId,
		uint8_t *&remainingResourcesCapacity, const uint16_t& precTime, int32_t activityDuration, const uint32_t& UBTime) {
	int32_t loadTime = 0, t = UBTime;
	for (t = precTime; t < UBTime && loadTime < activityDuration; ++t)       {
		bool capacityAvailable = true;
		for (int32_t resourceId = 0; resourceId < numberOfResources && capacityAvailable; ++resourceId)        {
			uint8_t activityRequirement = tex1Dfetch(cudaActivitiesResourcesTex, activityId*numberOfResources+resourceId);
			if (remainingResourcesCapacity[resourceId*UBTime+t] < activityRequirement)	{
				loadTime = 0;
				capacityAvailable = false;
			}
		}
		if (capacityAvailable == true)
			++loadTime;
	}
	return (uint16_t) t-loadTime;
}

/*!
 * \param activityId Identification of the added activity.
 * \param activityStart Scheduled start time of the activity.
 * \param activityStop Scheduled finish time of the activity.
 * \param numberOfResources Number of renewable resources in the project.
 * \param remainingResourcesCapacity Free capacity of each resource with respect to time.
 * \param UBTime Upper bound of the maximal duration of the project.
 * \brief It updates the state of all resources after activity is added.
 */
inline __device__ void cudaAddActivity(const uint16_t& activityId, const uint16_t& activityStart, const uint16_t& activityStop,
		const uint16_t& numberOfResources, uint8_t *&remainingResourcesCapacity, const uint32_t& UBTime)	{
	for (int32_t resourceId = 0; resourceId < numberOfResources; ++resourceId)     {
		uint8_t activityRequirement = tex1Dfetch(cudaActivitiesResourcesTex, activityId*numberOfResources+resourceId);
		for (uint32_t t = activityStart; t < activityStop; ++t)
			remainingResourcesCapacity[resourceId*UBTime+t] -= activityRequirement;
	}
}

/* CUDA IMPLEMENTATION OF THE BASE RESOURCE EVALUATION FUNCTIONS */

/*!
 * \param cudaData RCPSP constants, variables, ...
 * \param blockOrder Current order of the activities.
 * \param indexI Swap index i.
 * \param indexJ Swap index j.
 * \param activitiesDuration Duration of the activities.
 * \param resourceIndices Access indices for resources.
 * \param resourcesLoad Array of the earliest resource start times.
 * \param startValues Helper array for resource evaluation.
 * \param remainingResourcesCapacity Free capacity of each resource with respect to time.
 * \param startTimesWriterById Array of start times of the scheduled activities ordered by ID's.
 * \param capacityResolution If true then capacity based algorithm is selected else time based algorithm is selected.
 * \param forward It determines if schedule is forward or backward evaluated.
 * \return Schedule length without any penalties.
 * \brief Function evaluate schedule and return total schedule length.
 */
__device__ uint16_t cudaEvaluateOrder(const CudaData& cudaData, uint16_t *&blockOrder, const uint16_t& indexI, const uint16_t& indexJ, uint8_t *&activitiesDuration, uint16_t *&resourceIndices,
		uint16_t *resourcesLoad, uint16_t *startValues, uint8_t *remainingResourcesCapacity, uint16_t *startTimesWriterById, bool capacityResolution, bool forward = true)	{

	// Current cost of the schedule.
	uint16_t scheduleLength = 0;

	// Init state of resources.
	if (capacityResolution == true)
		cudaPrepareArrays(cudaData, resourcesLoad, startValues);
	else
		cudaPrepareArrays(cudaData.numberOfActivities, cudaData.numberOfResources, MAXIMAL_SUM_OF_FLOATS, remainingResourcesCapacity, resourceIndices);
	
	for (uint16_t i = 0; i < cudaData.numberOfActivities; ++i)	{

		uint16_t index = ((forward == true) ? i : cudaData.numberOfActivities-i-1);
		uint16_t activityId = blockOrder[index];

		// Logical swap.
		if (index == indexI)
			activityId = blockOrder[indexJ];

		if (index == indexJ)
			activityId = blockOrder[indexI];

		// Get the earliest start time without precedence penalty. (if moves are precedence penalty free)
		uint16_t start = 0;
		uint16_t baseIndex;
		uint16_t numberOfRelatedActivities;
		if (forward == true) {
			baseIndex = tex1Dfetch(cudaPredecessorsIndicesTex, activityId);
			numberOfRelatedActivities = tex1Dfetch(cudaPredecessorsIndicesTex, activityId+1)-baseIndex;
		} else	{
			baseIndex = tex1Dfetch(cudaSuccessorsIndicesTex, activityId);
			numberOfRelatedActivities = tex1Dfetch(cudaSuccessorsIndicesTex, activityId+1)-baseIndex;
		}
		for (uint16_t j = 0; j < numberOfRelatedActivities; ++j)	{
			uint16_t relatedActivityId;
			if (forward == true)
				relatedActivityId = tex1Dfetch(cudaPredecessorsTex, baseIndex+j);
			else
				relatedActivityId = tex1Dfetch(cudaSuccessorsTex, baseIndex+j);
			start = max(startTimesWriterById[relatedActivityId]+activitiesDuration[relatedActivityId], start);
		}

		// Get the earliest start time if the resources restrictions are counted.
		if (capacityResolution == true)
			start = max(cudaGetEarliestStartTime(cudaData.numberOfResources, activityId, resourcesLoad, resourceIndices), start);
		else
			start = max(cudaGetEarliestStartTime(cudaData.numberOfResources, activityId, remainingResourcesCapacity,
						start, activitiesDuration[activityId], MAXIMAL_SUM_OF_FLOATS), start);

		// Add activity = update resources arrays + write start time.
		uint16_t stop = start+activitiesDuration[activityId];
		if (capacityResolution == true)
			cudaAddActivity(activityId, start, stop, cudaData.numberOfResources, resourceIndices, resourcesLoad, startValues);
		else
			cudaAddActivity(activityId, start, stop, cudaData.numberOfResources, remainingResourcesCapacity, MAXIMAL_SUM_OF_FLOATS);
		scheduleLength = max(scheduleLength, stop);

		startTimesWriterById[activityId] = start;
	}

	return scheduleLength;
}

/*!
 * \param order Order of activities.
 * \param timeValuesById Time values of activities. Accessed through the identifications of activities.
 * \param size Length of the order and timeValuesById arrays.
 * \brief It reorders input order in accordance with timeValuesById array. It's stable sort with algorithm complexity O(n^2).
 */
inline __device__ void cudaInsertSort(uint16_t* order, const uint16_t * const& timeValuesById, const int16_t& size)	{
	for (int32_t i = 1; i < size; ++i)	{
		for (int32_t j = i; (j > 0) && ((timeValuesById[order[j]] < timeValuesById[order[j-1]]) == true); --j)	{
			uint16_t t = order[j];
			order[j] = order[j-1];
			order[j-1] = t;
		}
	}
}

/*!
 * \param order Order of activities.
 * \param startTimesById Start time values of activities. Accessed through the identifications of activities.
 * \param size Length of the order and timeValuesById arrays.
 * \brief It converts startTimesById array to activities order.
 */
inline __device__ void cudaConvertStartTimesById2ActivitiesOrder(uint16_t *& order, uint16_t *startTimesById, uint16_t size)	{
	cudaInsertSort(order, startTimesById, size);
}

/* CHECK PRECEDENCE FUNCTIONS */

/*!
 * \param successorsMatrix Bit matrix of successors.
 * \param numberOfActivities Number of activities.
 * \param activityId1 Activity identification.
 * \param activityId2 Activity identification.
 * \return True if an activity with identification activityId2 is successor of an activity with identification activityId1.
 * \brief Check if activity ID2 is successor of activity ID1.
 */
inline __device__ bool cudaGetMatrixBit(const uint8_t * const& successorsMatrix, const uint16_t& numberOfActivities, const int16_t& activityId1, const int16_t& activityId2)	{
	uint32_t bitPossition = activityId1*numberOfActivities+activityId2;
	if ((successorsMatrix[bitPossition/8] & (1<<(bitPossition % 8))) > 0)
		return true;
	else
		return false;
}

/*!
 * \param order Sequence of activities.
 * \param successorsMatrix Bit matrix of successors.
 * \param numberOfActivities Number of activities.
 * \param i Index i of swap.
 * \param j Index j of swap.
 * \param light If true then light version is executed. (precedences from activity at index i aren't checked)
 * \return True if current swap won't break relation precedences else false.
 * \brief Check if requested move is precedence penalty free.
 */
__device__ bool cudaCheckSwapPrecedencePenalty(const uint16_t * const& order, const uint8_t * const& successorsMatrix, const uint16_t& numberOfActivities, int16_t i, int16_t j, bool light = false)	{
	if (i > j)	{
		int16_t t = i;
		i = j; j = t;
	}
	for (uint16_t k = i; k < j; ++k)	{
		if (cudaGetMatrixBit(successorsMatrix, numberOfActivities, order[k], order[j]) == true)
			return false;
	}
	if (!light)	{
		for (uint16_t k = i+1; k <= j; ++k)	{
			if (cudaGetMatrixBit(successorsMatrix, numberOfActivities, order[i], order[k]) == true)
				return false;
		}
	}
	return true;
}

/*!
 * \param numAct The number of activities.
 * \param successorsMatrix Binary matrix of successors.
 * \param activitiesDuration Duration of each activity.
 * \param startTimesById Array of start time values of the scheduled activities ordered by ID's.
 * \return The precedence penalty.
 * \brief It finds out all precedence penalties and computes penalty.
 * \note The penalty should be zero since only non-precedence breaking moves are allowed.
 */
__device__ uint32_t cudaComputePrecedencePenalty(uint16_t numAct, uint8_t *successorsMatrix, uint8_t *activitiesDuration, uint16_t *startTimesById)  {
	uint32_t penalty = 0;
	for (uint16_t id1 = 0; id1 < numAct; ++id1)        {
		for (uint16_t id2 = 0; id2 < numAct; ++id2)        {
			if (id1 != id2 && cudaGetMatrixBit(successorsMatrix, numAct, id1, id2) == true)	{
				if (startTimesById[id1]+activitiesDuration[id1] > startTimesById[id2])
					penalty += startTimesById[id1]+activitiesDuration[id1]-startTimesById[id2];
			}

		}
	}
	return penalty;
}

/* SOFT VIOLATION PENALTIES */
#include <cstdio>
/*!
 * \param numberOfActivities The number of the activities in the project.
 * \param activitiesDuration Duration of each activity.
 * \param makespan The best known project makespan.
 * \param startTimesById Array of start time values of the scheduled activities ordered by ID's.
 * \return It returns overall tardiness penalty.
 */
__device__ uint32_t cudaComputeTardinessPenalty(uint16_t numberOfActivities, uint8_t *activitiesDuration, uint32_t makespan, uint16_t *startTimesById)	{
	uint32_t overhangPenalty = 0;
	for (uint16_t id = 0; id < numberOfActivities; ++id)	{
		if (startTimesById[id]+activitiesDuration[id]+rightLeftLongestPaths[id] > makespan)
			overhangPenalty += startTimesById[id]+activitiesDuration[id]+rightLeftLongestPaths[id]-makespan;
	}
	return overhangPenalty;
}

/*!
 * \param cudaData RCPSP constants, variables and data.
 * \param addedEdges Extra edges added to each solution in the solution set.
 * \param startTimesById Array of start time values computed by the evaluation algorithm.
 * \return The precedence penalty.
 */
__device__ uint32_t cudaComputePenaltyOfEdgeViolations(const CudaData& cudaData, Edge *& addedEdges, uint16_t *startTimesById)	{
	uint32_t precedencePenalty = 0;
	for (uint32_t e = 0; e < cudaData.numberOfAddedEdges; ++e)	{
		if (startTimesById[addedEdges[e].i]+addedEdges[e].weight > startTimesById[addedEdges[e].j])
			precedencePenalty += startTimesById[addedEdges[e].i]+addedEdges[e].weight-startTimesById[addedEdges[e].j];
	}
	if (precedencePenalty > 1000)
		printf("penalty: %d\n", precedencePenalty);
	return precedencePenalty;
}

/*!
 * \param cudaData RCPSP constants, variables, ...
 * \param blockOrder Order of activities.
 * \param bestScheduleStartTimesById Start time values of activities for the best shaked schedule.
 * \param activitiesDuration Duration of each activity.
 * \param resourceIndices Access indices for resources.
 * \param resourcesLoad Array of the earliest resource start times.
 * \param startValues Helper array for resource evaluation.
 * \param remainingResourcesCapacity Free capacity of each resource with respect to time.
 * \return The makespan of the best found shaked schedule.
 * \brief Iterative method tries to shake down activities in the schedule to ensure equally loaded resources. 
 * Therefore, the shorter schedule could be found.
 */
__device__ uint16_t cudaShakingDownEvaluation(const CudaData& cudaData, uint16_t *& blockOrder, uint16_t *bestScheduleStartTimesById, uint8_t *& activitiesDuration, uint16_t *& resourceIndices,
		uint16_t *resourcesLoad, uint16_t *startValues, uint8_t *remainingResourcesCapacity)	{

	uint16_t bestScheduleLength = 0xffff;
	uint16_t *currentOrder = new uint16_t[cudaData.numberOfActivities];
	if (!currentOrder)
		return bestScheduleLength;
	uint16_t *timeValuesById = new uint16_t[cudaData.numberOfActivities];
	if (!timeValuesById)	{
		delete[] currentOrder;
		return bestScheduleLength;
	}

	for (uint16_t i = 0; i < cudaData.numberOfActivities; ++i)
		currentOrder[i] = blockOrder[i];

	while (true)	{
		int32_t scheduleLength = cudaEvaluateOrder(cudaData, currentOrder, 0xffff, 0xffff, activitiesDuration, resourceIndices,
			       	resourcesLoad, startValues, remainingResourcesCapacity, timeValuesById, false, true);

		if (scheduleLength < bestScheduleLength)	{
			bestScheduleLength = scheduleLength;
			if (bestScheduleStartTimesById != NULL)	{
				for (uint16_t id = 0; id < cudaData.numberOfActivities; ++id)
					bestScheduleStartTimesById[id] = timeValuesById[id];
			}
		} else {
			break;
		}

		for (uint16_t id = 0; id < cudaData.numberOfActivities; ++id)
			timeValuesById[id] += activitiesDuration[id];

		cudaInsertSort(currentOrder, timeValuesById, cudaData.numberOfActivities);

		int32_t scheduleLengthBackward = cudaEvaluateOrder(cudaData, currentOrder, 0xffff, 0xffff, activitiesDuration,
			       	resourceIndices, resourcesLoad, startValues, remainingResourcesCapacity, timeValuesById, false, false);
		int32_t diffCmax = scheduleLength-scheduleLengthBackward;

		for (uint32_t id = 0; id < cudaData.numberOfActivities; ++id)
			timeValuesById[id] = scheduleLengthBackward-timeValuesById[id]-activitiesDuration[id];

		for (uint32_t id = 0; id < cudaData.numberOfActivities; ++id)	{
			if (((int32_t) timeValuesById[id])+diffCmax > 0)
				timeValuesById[id] += diffCmax;
			else
				timeValuesById[id] = 0;
		}

		cudaInsertSort(currentOrder, timeValuesById, cudaData.numberOfActivities);
	}

	delete[] timeValuesById;
	delete[] currentOrder;

	return bestScheduleLength;
}


/*	CUDA IMPLEMENT OF SIMPLE TABU LIST */

/*!
 * \param numberOfActivities Number of activities.
 * \param i Swap index i.
 * \param j Swap index j.
 * \param tabuCache Block tabu cache - fast check if move is in tabu list.
 * \return True if move is possible else false.
 * \brief Check if move is in tabu list.
 */
inline __device__ bool cudaIsPossibleMove(const uint16_t& numberOfActivities, const uint16_t& i, const uint16_t& j, uint8_t *&tabuCache)	{
	if (tabuCache[i*numberOfActivities+j] == 0 || tabuCache[j*numberOfActivities+i] == 0)
		return true;
	else
		return false;
}

/*!
 * \param numberOfActivities Number of activities.
 * \param i Swap index i of added move.
 * \param j Swap index j of added move.
 * \param tabuList Tabu list.
 * \param tabuCache Tabu cache.
 * \param tabuIdx Current index at tabu list.
 * \param tabuListSize Tabu list size.
 * \brief Add specified move to tabu list and update tabu cache.
 */
inline __device__ void cudaAddTurnToTabuList(const uint16_t& numberOfActivities, const uint16_t& i, const uint16_t& j, MoveIndices *&tabuList, uint8_t *&tabuCache, uint16_t& tabuIdx, const uint16_t& tabuListSize)	{

	MoveIndices move = tabuList[tabuIdx];
	uint16_t iOld = move.i, jOld = move.j;

	if (iOld != 0 && jOld != 0)
		tabuCache[iOld*numberOfActivities+jOld] = tabuCache[jOld*numberOfActivities+iOld] = 0;

	move.i = i; move.j = j;
	tabuList[tabuIdx] = move;
	tabuCache[i*numberOfActivities+j] = tabuCache[j*numberOfActivities+i] = 1;

	tabuIdx = (tabuIdx+1) % tabuListSize;
}


/* HELP FUNCTIONS */

/*!
 * \param numberOfActivities Number of activities.
 * \param tabuList Tabu list.
 * \param tabuCache Tabu cache.
 * \param numberOfElements Number of tabu list elements that will be removed.
 * \brief Remove specified number of elements from tabu list and update tabu cache.
 */
inline __device__ void cudaClearTabuList(const uint16_t& numberOfActivities, MoveIndices *tabuList, uint8_t *tabuCache, const uint16_t& numberOfElements)	{
	for (uint16_t k = threadIdx.x; k < numberOfElements; k += blockDim.x)	{
		MoveIndices *tabuMove = &tabuList[k];
		uint16_t i = tabuMove->i, j = tabuMove->j;
		tabuCache[i*numberOfActivities+j] = tabuCache[j*numberOfActivities+i] = 0;
		tabuMove->j = tabuMove->i = 0;
	}
	__syncthreads();
	return;
}

/*!
 * \param numberOfActivities Number of activities.
 * \param tabuList Tabu list.
 * \param tabuCache Tabu cache.
 * \param tabuListSize Block tabu list size.
 * \param blockOrder Block schedule - order.
 * \param externalSolution Solution from a set or the best global solution. (order)
 * \param externalTabuList Tabu list of external solution.
 * \brief Replace current block solution with a read external solution (order+tabu).
 */
inline __device__ void cudaReadExternalSolution(const uint16_t& numberOfActivities, MoveIndices *tabuList, uint8_t *tabuCache, const uint16_t& tabuListSize,
		uint16_t *blockOrder, uint16_t *externalSolution, MoveIndices *externalTabuList)	{
	// Clear current tabu list and tabu cache.
	cudaClearTabuList(numberOfActivities, tabuList, tabuCache, tabuListSize);
	// Read block order.
	for (uint16_t i = threadIdx.x; i < numberOfActivities; i += blockDim.x)
		blockOrder[i] = externalSolution[i];
	// Read block tabu list and create tabu cache.
	for (uint16_t l = threadIdx.x; l < tabuListSize; l += blockDim.x)	{
		tabuList[l] = externalTabuList[l];
		MoveIndices *move = &tabuList[l];
		uint16_t i = move->i, j = move->j;
		tabuCache[i*numberOfActivities+j] = tabuCache[j*numberOfActivities+i] = 1;
	}
	__syncthreads();
	return;
}

/* REORDER ARRAY FUNCTION */

/*!
 * \tparam T uint16_t or uint32_t.
 * \param moves Array of moves which should be reorder.
 * \param resultMerge Result array of reordered moves.
 * \param threadsCounter Helper array for threads counters.
 * \param size How many elements will be processed at moves array.
 * \return Number of written elements to resultMerge array.
 * \brief Move all valid moves to the resultMerge array and return number of valid moves.
 */
template <typename T>
inline __device__ uint32_t cudaReorderMoves(uint32_t *moves, uint32_t *resultMerge, T *threadsCounter, const uint32_t& size)	{
	threadsCounter[threadIdx.x] = 0;
	uint32_t threadAmount = size/blockDim.x+1;
	for (uint32_t i = threadIdx.x*threadAmount; i < size && i < (threadIdx.x+1)*threadAmount; ++i)	{
		if (moves[i] != 0)
			++threadsCounter[threadIdx.x];
	}
	__syncthreads();
	for (uint32_t k = 0; (1<<k) < blockDim.x; ++k)   {
		uint32_t step = 1<<k;
		uint32_t begIdx = (step-1)+2*step*threadIdx.x;
		if (begIdx < blockDim.x-step)
			threadsCounter[begIdx+step] += threadsCounter[begIdx];
		__syncthreads();
	}
	for (int32_t k = (blockDim.x>>1); k > 1; k >>= 1)	{
		uint32_t step = k/2;
		uint32_t begIdx = (k-1)+2*step*threadIdx.x;
		if (begIdx < blockDim.x-step) 
			threadsCounter[begIdx+step] += threadsCounter[begIdx];
		__syncthreads();
	}
	uint32_t threadStartIndex = threadIdx.x > 0 ? threadsCounter[threadIdx.x-1] : 0;
	for (uint32_t i = threadIdx.x*threadAmount; i < size && i < (threadIdx.x+1)*threadAmount; ++i)	{
		if (moves[i] != 0)
			resultMerge[threadStartIndex++] = moves[i];
	}
	__syncthreads();
	return threadsCounter[blockDim.x-1];
}

/* DIVERSIFICATION FUNCTION */

/*!
 * \param numberOfActivities Number of activities.
 * \param order Current schedule - sequence of activities.
 * \param successorsMatrix Bit matrix of successors.
 * \param diversificationSwaps Number of diversification swaps.
 * \param state State of the random generator.
 * \brief Function performs specified number of precedence penalty free swaps.
 */
inline __device__ void cudaDiversificationOfSolution(const uint16_t& numberOfActivities, uint16_t *order, const uint8_t *successorsMatrix, const uint32_t& diversificationSwaps, curandState *state)	{
		
	uint32_t performedSwaps = 0;
	while (performedSwaps < diversificationSwaps)  {
		uint16_t i = (curand(state) % (numberOfActivities-2)) + 1;
		uint16_t j = (curand(state) % (numberOfActivities-2)) + 1;
		if ((i != j) && (cudaCheckSwapPrecedencePenalty(order, successorsMatrix, numberOfActivities, i, j) == true))	{
			uint16_t t = order[i];
			order[i] = order[j];
			order[j] = t;
			++performedSwaps;
		}
	}
	return;
}


/*	CUDA IMPLEMENT OF GLOBAL KERNEL */

/*!
 * Global function for RCPSP problem. Blocks communicate with each other through global memory.
 * Local variables are coalesced. Dynamic shared memory and texture memory is used.
 * \param cudaData All required constants, pointers to device memory, setting variables, ....
 * \brief Solve RCPSP problem on GPU.
 */
__global__ void cudaSolveRCPSP(const CudaData cudaData)	{
	
	__shared__ bool initialized;
	__shared__ uint32_t iter;
	__shared__ MoveInfo iterBestMove;
	__shared__ Edge *blockAddedEdges;
	__shared__ uint32_t blockBestCost;
	__shared__ uint16_t *blockBestSolution;
	__shared__ uint32_t maximalNeighbourhoodSize;
	__shared__ uint8_t *blockActivitiesDuration;
	__shared__ uint16_t *blockCurrentOrder;
	__shared__ uint8_t *blockSuccessorsMatrix;
	__shared__ MoveInfo *blockMergeArray;
	__shared__ uint16_t *blockPartitionCounterUInt16;
	__shared__ uint32_t *blockPartitionCounterUInt32;
	__shared__ MoveIndices *blockReorderingArray;
	__shared__ MoveIndices *blockReorderingArrayHelp;

	__shared__ uint16_t blockTabuIdx;
	__shared__ uint16_t blockTabuListSize;
	__shared__ MoveIndices *blockTabuList;
	__shared__ uint8_t *blockTabuCache;
	__shared__ int32_t blockIndexOfSetSolution;
	__shared__ bool blockReadPossible;
	__shared__ bool blockWriteBestBlock;
	__shared__ bool blockReadSetSolution;
	__shared__ bool blockWriteSetSolution;
	__shared__ bool blockCriticalPathLengthAchieved;
	__shared__ uint32_t blockNumberOfIterationsSinceBest;
	__shared__ uint32_t blockMaximalNumberOfIterationsSinceBest;
	__shared__ uint16_t *blockResourceIndices;

	__shared__ curandState randState;

	curandState threadRandState;
	curand_init(blockDim.x*blockIdx.x+threadIdx.x, threadIdx.x, 0, &threadRandState);

	uint16_t threadResourcesLoad[TOTAL_SUM_OF_CAPACITY];
	uint16_t threadStartValues[MAXIMUM_CAPACITY_OF_RESOURCE];
	uint8_t threadRemainingResourcesCapacity[NUMBER_OF_RESOURCES*MAXIMAL_SUM_OF_FLOATS];
	uint16_t threadStartTimesById[NUMBER_OF_ACTIVITIES];

	extern __shared__ uint8_t dynamicSharedMemory[];
	if (threadIdx.x == 0)	{
		/* SET VARIABLES */
		iter = 0;
		initialized = false;
		blockTabuIdx = 0;
		blockWriteBestBlock = false;
		blockReadSetSolution = false;
		blockWriteSetSolution = false;
		blockCriticalPathLengthAchieved= false;
		blockNumberOfIterationsSinceBest = 0;
		blockIndexOfSetSolution = blockIdx.x % cudaData.totalSolutions;
		maximalNeighbourhoodSize = (cudaData.numberOfActivities-2)*cudaData.swapRange;
		blockReorderingArray = cudaData.swapMergeArray+blockIdx.x*maximalNeighbourhoodSize;
		blockReorderingArrayHelp = cudaData.mergeHelpArray+blockIdx.x*maximalNeighbourhoodSize;
		blockTabuList = cudaData.tabuLists+blockIdx.x*cudaData.maxTabuListSize;
		blockTabuListSize = cudaData.maxTabuListSize-((cudaData.maxTabuListSize*blockIdx.x)/(4*gridDim.x));
		blockTabuCache = cudaData.tabuCaches+blockIdx.x*cudaData.numberOfActivities*cudaData.numberOfActivities;
		blockBestSolution = cudaData.blocksBestSolution+blockIdx.x*cudaData.numberOfActivities;

		curand_init(3*blockIdx.x+71, blockIdx.x, 0, &randState);
		blockMaximalNumberOfIterationsSinceBest = curand(&randState) % cudaData.maximalIterationsSinceBest;
		
		/* ASSIGN SHARED MEMORY */
		blockMergeArray = (MoveInfo*) dynamicSharedMemory; 
		blockAddedEdges = (Edge*) (blockMergeArray+blockDim.x);
		if (maximalNeighbourhoodSize < 0xffff)	{
			blockPartitionCounterUInt16 = (uint16_t*) (blockAddedEdges+cudaData.numberOfAddedEdges);
			blockPartitionCounterUInt32 = NULL;
			blockCurrentOrder = blockPartitionCounterUInt16+blockDim.x;
		} else	{
			blockPartitionCounterUInt32 = (uint32_t*) (blockAddedEdges+cudaData.numberOfAddedEdges);
			blockPartitionCounterUInt16 = NULL;
			blockCurrentOrder = (uint16_t*) (blockPartitionCounterUInt32+blockDim.x);
		}	
		blockResourceIndices = blockCurrentOrder+cudaData.numberOfActivities;
		blockActivitiesDuration = (uint8_t*) (blockResourceIndices+cudaData.numberOfResources+1);
		if (cudaData.copySuccessorsMatrixToSharedMemory)
			blockSuccessorsMatrix = blockActivitiesDuration+cudaData.numberOfActivities;
		else
			blockSuccessorsMatrix = cudaData.successorsMatrix;
	}
	__syncthreads();

	for (uint32_t i = threadIdx.x; i < cudaData.numberOfResources+1; i += blockDim.x)	{
		blockResourceIndices[i] = cudaData.resourceIndices[i];
	}

	for (uint32_t i = threadIdx.x; i < cudaData.numberOfActivities; i += blockDim.x)	{
		blockActivitiesDuration[i] = cudaData.durationOfActivities[i];
	}

	for (uint32_t i = threadIdx.x; i < cudaData.numberOfAddedEdges; i += blockDim.x)	{
		blockAddedEdges[i] = cudaData.addedEdges[blockIndexOfSetSolution*cudaData.numberOfAddedEdges+i];
	}

	if (cudaData.copySuccessorsMatrixToSharedMemory)	{
		for (uint32_t i = threadIdx.x; i < cudaData.successorsMatrixSize; i += blockDim.x)
			blockSuccessorsMatrix[i] = cudaData.successorsMatrix[i];
	}

	// Block have to obtain initial read access.
	if (threadIdx.x == 0)	{
		while (atomicCAS(cudaData.lockSetOfSolutions, DATA_AVAILABLE, DATA_ACCESS) != DATA_AVAILABLE)
			;
		blockBestCost = cudaData.infoAboutSolutions[blockIndexOfSetSolution].solutionCost;
	}
	__syncthreads();

	// Copy solution from a set of solutions to local block order.
	for (uint32_t i = threadIdx.x; i < cudaData.numberOfActivities; i += blockDim.x)	{
		blockCurrentOrder[i] = cudaData.ordersOfSolutions[blockIndexOfSetSolution*cudaData.numberOfActivities+i];
	}
	__syncthreads();

	// Free read lock.
	if (threadIdx.x == 0)	{
		atomicExch(cudaData.lockSetOfSolutions, DATA_AVAILABLE);
	}

	while (iter < cudaData.numberOfIterationsPerBlock && !blockCriticalPathLengthAchieved)	{

		for (uint16_t i = threadIdx.x+1; i < (cudaData.numberOfActivities-1); i += blockDim.x)	{
			bool relationsBroken = false;
			struct MoveIndices info;
			for (uint16_t j = i+1; j < i+1+cudaData.swapRange; ++j)	{
				info.i = info.j = 0;
				if ((i < cudaData.numberOfActivities-2) && (j < cudaData.numberOfActivities-1) && !relationsBroken)	{
					if (cudaGetMatrixBit(blockSuccessorsMatrix, cudaData.numberOfActivities, blockCurrentOrder[i], blockCurrentOrder[j]) == false)	{
						info.i = i; info.j = j;
					}	else	{
						relationsBroken = true;
					}
				}
				blockReorderingArray[(i-1)*cudaData.swapRange+(j-1-i)] = info;
			}
		}
		__syncthreads();

		uint32_t swapMoves = 0;
		if (blockPartitionCounterUInt16 != NULL)
			swapMoves = cudaReorderMoves((uint32_t*) blockReorderingArray, (uint32_t*) blockReorderingArrayHelp,  blockPartitionCounterUInt16, maximalNeighbourhoodSize);
		else
			swapMoves = cudaReorderMoves((uint32_t*) blockReorderingArray, (uint32_t*) blockReorderingArrayHelp,  blockPartitionCounterUInt32, maximalNeighbourhoodSize);

		for (uint32_t i = threadIdx.x; i < swapMoves; i += blockDim.x)	{
			struct MoveIndices *move = &blockReorderingArrayHelp[i];
			if (cudaCheckSwapPrecedencePenalty(blockCurrentOrder, blockSuccessorsMatrix, cudaData.numberOfActivities, move->i, move->j, true) == false)	{
				move->i = move->j = 0;
			}
		}
		__syncthreads();
		
		if (blockPartitionCounterUInt16 != NULL)
			swapMoves = cudaReorderMoves((uint32_t*) blockReorderingArrayHelp, (uint32_t*) blockReorderingArray,  blockPartitionCounterUInt16, swapMoves);
		else
			swapMoves = cudaReorderMoves((uint32_t*) blockReorderingArrayHelp, (uint32_t*) blockReorderingArray,  blockPartitionCounterUInt32, swapMoves);

		blockMergeArray[threadIdx.x].cost = 0xffffffff;
		for (uint32_t i = threadIdx.x; i < swapMoves; i += blockDim.x)	{
			struct MoveIndices *move = &blockReorderingArray[i];
			uint32_t threadBestCost = blockMergeArray[threadIdx.x].cost;
			uint32_t totalEval = cudaEvaluateOrder(cudaData, blockCurrentOrder, move->i, move->j, blockActivitiesDuration, blockResourceIndices, threadResourcesLoad,
					threadStartValues, threadRemainingResourcesCapacity, threadStartTimesById, cudaData.capacityResolutionAlgorithm);
			totalEval += cudaComputeTardinessPenalty(cudaData.numberOfActivities, blockActivitiesDuration, blockBestCost-1, threadStartTimesById);
			totalEval += cudaComputePenaltyOfEdgeViolations(cudaData, blockAddedEdges, threadStartTimesById);
			totalEval = (totalEval > 0x0000ffff ? 0xffff0000 : totalEval<<16);
			totalEval |= (curand(&threadRandState) & 0x0000ffff);
			uint32_t precedencePenalty = cudaComputePrecedencePenalty(cudaData.numberOfActivities, blockSuccessorsMatrix, blockActivitiesDuration, threadStartTimesById);
			if (precedencePenalty > 0)	{
				printf("ERROR: block %d, thread %d, infeasible solution!\n", blockIdx.x, threadIdx.x);
			}
			bool isPossibleMove = cudaIsPossibleMove(cudaData.numberOfActivities, move->i, move->j, blockTabuCache);
			if ((isPossibleMove && totalEval < threadBestCost) || (totalEval>>16) < blockBestCost)	{
				struct MoveInfo newBestThreadSolution = { move->i, move->j, totalEval };
				blockMergeArray[threadIdx.x] = newBestThreadSolution;
			}
		}
		__syncthreads();

		for (uint16_t k = blockDim.x/2; k > 0; k >>= 1)	{
			if (threadIdx.x < k)	{
				if (blockMergeArray[threadIdx.x].cost > blockMergeArray[threadIdx.x+k].cost)
					blockMergeArray[threadIdx.x] = blockMergeArray[threadIdx.x+k];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)	{
			blockReadPossible = false;
			iterBestMove = blockMergeArray[0];
			iterBestMove.cost >>= 16;
			atomicAdd((unsigned long long*) cudaData.evaluatedSchedules, swapMoves);
			atomicInc(&cudaData.infoAboutSolutions[blockIndexOfSetSolution].iterationCounter, 0xffffffff);
			if (iterBestMove.cost < blockBestCost)	{
				blockWriteBestBlock = true;
				blockBestCost = iterBestMove.cost;
				blockNumberOfIterationsSinceBest = 0;
			}

			uint32_t readSlotCost = cudaData.infoAboutSolutions[blockIndexOfSetSolution].solutionCost;
			if (blockNumberOfIterationsSinceBest >= blockMaximalNumberOfIterationsSinceBest || readSlotCost != blockBestCost || *cudaData.bestSolutionCost == cudaData.criticalPathLength) {
				bool setOfSolutionsAccess = false;
				if (atomicCAS(cudaData.lockSetOfSolutions, DATA_AVAILABLE, DATA_ACCESS) == DATA_AVAILABLE)
					setOfSolutionsAccess = true;

				if (setOfSolutionsAccess)	{

					if (blockBestCost < cudaData.infoAboutSolutions[blockIndexOfSetSolution].solutionCost)	{
						blockWriteSetSolution = true;
						cudaData.infoAboutSolutions[blockIndexOfSetSolution].readCounter = 0;
						cudaData.infoAboutSolutions[blockIndexOfSetSolution].solutionCost = blockBestCost;
						if (blockBestCost < *cudaData.bestSolutionCost)	{
							*cudaData.bestSolutionCost =  blockBestCost;
							*cudaData.indexToTheBestSolution = blockIndexOfSetSolution;
						}
					}	else	{
						atomicExch(cudaData.lockSetOfSolutions, DATA_AVAILABLE);
					}

					if (*cudaData.bestSolutionCost == cudaData.criticalPathLength)	{
						blockCriticalPathLengthAchieved = true;
					}

					if (readSlotCost < blockBestCost || blockNumberOfIterationsSinceBest >= blockMaximalNumberOfIterationsSinceBest)	{
						blockReadSetSolution = true;
					}
				}
			}  else if (!blockWriteBestBlock)	{
				++blockNumberOfIterationsSinceBest;
			}
		}
		
		if (blockMergeArray[0].cost == 0xffffffff)	{
			// Empty expanded neighborhood. Tabu list will be pruned.
			cudaClearTabuList(cudaData.numberOfActivities, blockTabuList, blockTabuCache, blockTabuListSize/3);
		} else if (threadIdx.x == 0)	{
			// Apply best move.
			uint16_t t = blockCurrentOrder[iterBestMove.i];
			blockCurrentOrder[iterBestMove.i] = blockCurrentOrder[iterBestMove.j];
			blockCurrentOrder[iterBestMove.j] = t;
			// Add move to tabu list.
			cudaAddTurnToTabuList(cudaData.numberOfActivities, iterBestMove.i, iterBestMove.j, blockTabuList, blockTabuCache, blockTabuIdx, blockTabuListSize);
		}
		__syncthreads();

		if (blockWriteBestBlock == true)	{
			if (threadIdx.x == 0)	{
				uint16_t improvedCost = cudaShakingDownEvaluation(cudaData, blockCurrentOrder, threadStartTimesById, blockActivitiesDuration, 
						blockResourceIndices, threadResourcesLoad, threadStartValues, threadRemainingResourcesCapacity);
				if (improvedCost < blockBestCost)	{
					blockBestCost = improvedCost;
					cudaConvertStartTimesById2ActivitiesOrder(blockCurrentOrder, threadStartTimesById, cudaData.numberOfActivities);
				}
			}
			__syncthreads();
			for (uint16_t i = threadIdx.x; i < cudaData.numberOfActivities; i += blockDim.x)
				blockBestSolution[i] = blockCurrentOrder[i];
			blockWriteBestBlock = false; initialized = true;
		}
		__syncthreads();

		if (blockWriteSetSolution == true)	{
			if (!initialized)
				printf("ERROR - write not-initialized solution!!\n");
			for (uint16_t i = threadIdx.x; i < cudaData.numberOfActivities; i += blockDim.x)
				cudaData.ordersOfSolutions[blockIndexOfSetSolution*cudaData.numberOfActivities+i] = blockBestSolution[i];
			for (uint16_t i = threadIdx.x; i < cudaData.maxTabuListSize; i += blockDim.x)
				cudaData.tabuListsOfSetOfSolutions[blockIndexOfSetSolution*cudaData.maxTabuListSize+i] = blockTabuList[i];
			__threadfence();
			__syncthreads();
			if (threadIdx.x == 0)	{
				blockWriteSetSolution = false; 
				atomicExch(cudaData.lockSetOfSolutions, DATA_AVAILABLE);
			}
		}

		if (blockReadSetSolution == true)	{
			if (threadIdx.x == 0)	{
				if (atomicCAS(cudaData.lockSetOfSolutions, DATA_AVAILABLE, DATA_ACCESS) == DATA_AVAILABLE)
					blockReadPossible = true;
			}
			__syncthreads();
			if (blockReadPossible)	{
				if (threadIdx.x == 0)	{
					blockIndexOfSetSolution = (blockIndexOfSetSolution+1) % cudaData.totalSolutions;
				}
				__syncthreads();
				// Read solution from a set to block memory.
				cudaReadExternalSolution(cudaData.numberOfActivities, blockTabuList, blockTabuCache, blockTabuListSize, blockCurrentOrder,
						cudaData.ordersOfSolutions+blockIndexOfSetSolution*cudaData.numberOfActivities, cudaData.tabuListsOfSetOfSolutions+blockIndexOfSetSolution*cudaData.maxTabuListSize);
				if (threadIdx.x == 0)	{
					blockBestCost = cudaData.infoAboutSolutions[blockIndexOfSetSolution].solutionCost;
					uint32_t readCounter = ++cudaData.infoAboutSolutions[blockIndexOfSetSolution].readCounter;
					blockNumberOfIterationsSinceBest = 0;

					blockReadSetSolution = false; initialized = false;
					blockMaximalNumberOfIterationsSinceBest = curand(&randState) % cudaData.maximalIterationsSinceBest;
					atomicExch(cudaData.lockSetOfSolutions, DATA_AVAILABLE);
					if (readCounter > cudaData.maximalValueOfReadCounter)
						cudaDiversificationOfSolution(cudaData.numberOfActivities, blockCurrentOrder, blockSuccessorsMatrix, cudaData.numberOfDiversificationSwaps, &randState);
				}
			}
		}

		if (threadIdx.x == 0)	{
			++iter;
		}
		__syncthreads();
	}

	// Write solution if is better than the best found.
	if (threadIdx.x == 0)	{
		while (atomicCAS(cudaData.lockSetOfSolutions, DATA_AVAILABLE, DATA_ACCESS) != DATA_AVAILABLE)
			;
	}
	__syncthreads();

	if (*cudaData.bestSolutionCost > blockBestCost)	{
		if (threadIdx.x == 0 && !initialized)	{
			printf("ERROR - write not-initialized solution!!\n");
			printf("original %d; new %d\n", *cudaData.bestSolutionCost, blockBestCost);
		}
		for (uint16_t i = threadIdx.x; i < cudaData.numberOfActivities; i += blockDim.x)
			cudaData.ordersOfSolutions[blockIndexOfSetSolution*cudaData.numberOfActivities+i] = blockBestSolution[i];
		for (uint16_t i = threadIdx.x; i < cudaData.maxTabuListSize; i += blockDim.x)
			cudaData.tabuListsOfSetOfSolutions[blockIndexOfSetSolution*cudaData.maxTabuListSize+i] = blockTabuList[i];
		if (threadIdx.x == 0)	{
			*cudaData.bestSolutionCost = blockBestCost;
			*cudaData.indexToTheBestSolution = blockIndexOfSetSolution;
		}
	}
	__threadfence();
	__syncthreads();

	if (threadIdx.x == 0)
		atomicExch(cudaData.lockSetOfSolutions, DATA_AVAILABLE);

	return;
}


/* START MAIN CUDA KERNEL */

void runCudaSolveRCPSP(int numberOfBlock, int numberOfThreadsPerBlock, int computeCapability, int dynSharedMemSize, const CudaData& cudaData)	{
	if (dynSharedMemSize < 7950)	{
		// Prefare 16 kB shared memory + 48 kB cache L1.
		cudaFuncSetCacheConfig(cudaSolveRCPSP, cudaFuncCachePreferL1);
	} else {
		// Prefare 48 kB shared memory + 16 kB cache L1.
		cudaFuncSetCacheConfig(cudaSolveRCPSP, cudaFuncCachePreferShared);
	}
	// Set maximum amount of dynamic memory to 1 MB.
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024);
	// Launch the main GPU kernel.
	cudaSolveRCPSP<<<numberOfBlock,numberOfThreadsPerBlock,dynSharedMemSize>>>(cudaData);
	cudaDeviceSynchronize();
}


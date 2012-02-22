
/*	CUDA IMPLEMENT OF SOURCES LOAD */

__device__ void cudaPrepareArrays(uint16_t *cudaResourcesLoad, uint16_t *cudaStartValues, uint8_t *cudaRequiredItems, uint16_t *cudaStartTimesById)	{
	for (uint16_t i = 0; i < TOTAL_SUM_OF_CAPACITY; ++i)	{
		cudaResourcesLoad[i] = 0;
		if (i < MAXIMUM_CAPACITY_OF_RESOURCE)	{
			cudaRequiredItems[i] = 0;
			cudaStartValues[i] = 0;
		}
	}
	for (uint16_t i = 0; i < NUMBER_OF_ACTIVITIES; ++i)
		cudaStartTimesById[i] = 0;
}

__device__ uint16_t cudaGetEarliestStartTime(uint16_t actId, uint8_t numRes, uint16_t *cudaResourcesLoad, uint16_t *cudaResourcesIdxs) {
	uint16_t bestStart = 0;
	for (uint8_t resourceId = 0; resourceId < numRes; ++resourceId)	{
		uint8_t activityRequirement = cudaActivitiesResources[actId*numRes+resourceId];
		if (activityRequirement > 0)
			bestStart = max(cudaResourcesLoad[cudaResourcesIdxs[resourceId]+activityRequirement-1], bestStart);
	}
	return bestStart;
}

__device__ void cudaAddActivity(uint16_t actId, uint16_t activityStart, uint16_t activityStop, uint8_t numRes, uint16_t *cudaResourcesLoad,  uint16_t *cudaResIdxs, uint16_t *cudaStartValues, uint8_t *cudaReqItems)	{
	bool writeValue, workDone;
	uint16_t sourceReq, curLoad, idx;
	for (uint8_t resourceId = 0; resourceId < numRes; ++resourceId)	{
		sourceReq = cudaActivitiesResources[actId*numRes+resourceId];
		for (uint8_t capIdx = cudaResIdxs[resourceId+1]-cudaResIdxs[resourceId]; capIdx > 0; --capIdx)	{
			curLoad = cudaResourcesLoad[cudaResIdxs[resourceId]+capIdx-1];
			if (sourceReq > 0)	{
				if (curLoad <= activityStart)	{
					cudaResourcesLoad[cudaResIdxs[resourceId]+capIdx-1] = activityStop;
					--sourceReq;
					idx = 0;
					while (cudaStartValues[idx] != 0)	{
						if (cudaReqItems[idx] > 0)
							--cudaReqItems[idx];
						++idx;
					}
				} else if (curLoad < activityStop)	{
					cudaResourcesLoad[cudaResIdxs[resourceId]+capIdx-1] = activityStop;
					--sourceReq;
					idx = 0;
					writeValue = true;
					while (cudaStartValues[idx] != 0)	{
						if (cudaStartValues[idx] > curLoad && cudaReqItems[idx] > 0)
							--cudaReqItems[idx];
						if (cudaStartValues[idx] == curLoad)
							writeValue = false;
						++idx;
					}
					if (writeValue == true && curLoad > 0)	{
						cudaStartValues[idx] = curLoad;
						cudaReqItems[idx] = cudaActivitiesResources[actId*numRes+resourceId];
					}
				}
			} else {
				idx = 0;
				workDone = true;
				while (cudaStartValues[idx] != 0)	{
					if (cudaReqItems[idx] != 0)	{
						workDone = false;
						break;
					}
					++idx;
				}
				if (workDone == true)	{
					break;
				} else {
					writeValue = true;
					cudaResourcesLoad[cudaResIdxs[resourceId]+capIdx-1] = cudaStartValues[idx];
					idx = 0;
					while (cudaStartValues[idx] != 0)	{
						if (curLoad < cudaStartValues[idx] && cudaReqItems[idx] > 0)
							--cudaReqItems[idx];
						if (curLoad == cudaStartValues[idx])
							writeValue = false;
						++idx;
					}
					if (writeValue == true && curLoad > activityStart)	{
						cudaStartValues[idx] = curLoad;
						cudaReqItems[idx] = cudaActivitiesResources[actId*numRes+resourceId];
					}
				}
			}
		}
		idx = 0;
		while (cudaStartValues[idx] != 0)
			cudaStartValues[idx++] = 0;
	}
}

/* CUDA IMPLEMENT OF BASE SCHEDULE SOLVER FUNCTIONS */


__device__ uint16_t cudaEvaluateOrder(uint16_t numAct, uint8_t numRes, uint16_t *cudaBlockOrder, uint16_t actX, uint16_t actY,
		uint16_t *cudaStartTimesWriterById, uint16_t *cudaResourcesLoad, uint16_t *cudaStartValues, uint8_t *cudaRequiredItems, uint16_t *cudaResIdxs, uint16_t *cudaPredIdxs)	{

	uint16_t start = 0, scheduleLength = 0;

	cudaPrepareArrays(cudaResourcesLoad, cudaStartValues, cudaRequiredItems, cudaStartTimesWriterById);
	
	for (uint16_t i = 0; i < numAct; ++i)	{

		uint16_t activityId = cudaBlockOrder[i];

		if (i == actX)
			activityId = cudaBlockOrder[actY];

		if (i == actY)
			activityId = cudaBlockOrder[actX];

		for (uint16_t j = 0; j < (cudaPredIdxs[activityId+1]-cudaPredIdxs[activityId]); ++j)	{
			uint16_t predecessorId = tex1Dfetch(cudaPredecessorsTex, cudaPredIdxs[activityId]+j);
			start = max(cudaStartTimesWriterById[predecessorId]+cudaActivitiesDuration[predecessorId], start);
		}

		start = max(cudaGetEarliestStartTime(activityId, numRes, cudaResourcesLoad, cudaResIdxs), start);
		cudaAddActivity(activityId, start, start+cudaActivitiesDuration[activityId], numRes, cudaResourcesLoad, cudaResIdxs, cudaStartValues, cudaRequiredItems);
		scheduleLength = max(scheduleLength, start+cudaActivitiesDuration[activityId]);

		cudaStartTimesWriterById[activityId] = start;
	}

	return scheduleLength;
}

__device__ uint32_t cudaComputePrecedencePenalty(uint16_t numAct, uint16_t *cudaSucIdxs, uint16_t *cudaStartTimesById)	{
	uint32_t penalty = 0;
	for (uint16_t activityId = 0; activityId < numAct; ++activityId)	{
		for (uint16_t j = 0; j < (cudaSucIdxs[activityId+1]-cudaSucIdxs[activityId]); ++j)	{
			uint16_t successorId = tex1Dfetch(cudaSuccessorsTex, cudaSucIdxs[activityId]+j);
			if (cudaStartTimesById[activityId]+cudaActivitiesDuration[activityId] > cudaStartTimesById[successorId])
				penalty += cudaStartTimesById[activityId]+cudaActivitiesDuration[activityId]-cudaStartTimesById[successorId];
		}
	}
	return PRECEDENCE_PENALTY*penalty;
}

__device__ uint32_t cudaComputeHashTableIndex(uint16_t numAct, uint16_t *cudaBlockOrder, uint16_t actX, uint16_t actY, uint32_t actI, uint32_t actJ)	{
	uint32_t hashValue = 1;
/*
	hashValue *= (R+2*actI);
	hashValue ^= actI;
*/
	for (uint32_t i = 1; i < numAct-1; ++i)	{
		uint32_t activityId = cudaBlockOrder[i];
		if (i == actX)
			activityId = cudaBlockOrder[actY];
		if (i == actY)
			activityId = cudaBlockOrder[actX];

		hashValue *= (R+2*activityId*i);
		hashValue ^= activityId;
	}
/*
	hashValue *= (R+2*actJ);
	hashValue ^= actJ;
*/
	hashValue /= 2;
	hashValue &= 0x00ffffff;	// Size of hash table is 2^24.

	return hashValue;
}

/*	CUDA IMPLEMENT OF SIMPLE TABU LIST */

__device__ bool cudaIsPossibleMove(uint16_t numAct, uint16_t i, uint16_t j, uint8_t *tabuCache)	{
	if (tabuCache[i*numAct+j] == 0 || tabuCache[j*numAct+i] == 0)
		return true;
	else
		return false;
}

__device__ void cudaAddTurnToTabuList(uint16_t numAct, uint16_t i, uint16_t j, uint32_t *tabuList, uint8_t *tabuCache, uint16_t& tabuIdx)	{
	uint32_t listEl = tabuList[tabuIdx];
	uint16_t iOld = (listEl>>16), jOld = (listEl & 0x0000ffff);
	if (iOld != USHRT_MAX && jOld != USHRT_MAX)	{
		tabuCache[iOld*numAct+jOld] = tabuCache[jOld*numAct+iOld] = 0;
	}

	tabuCache[i*numAct+j] = tabuCache[j*numAct+i] = 1;

	listEl = (((uint32_t) i)<<16)+j;
	tabuList[tabuIdx] = listEl;

	tabuIdx = (tabuIdx+1) % TABU_LIST_SIZE;
}

/*	CUDA IMPLEMENT OF GLOBAL KERNEL */
#include <cstdio>

__global__ void solveRCPSP(uint16_t numAct, uint8_t numRes, uint16_t *sucIdx, uint16_t *predIdx, uint32_t *bcost, uint16_t *border, uint32_t *tab, uint8_t *tabCache,
		uint16_t *capIdxs, uint16_t *startTimesId, uint32_t maxIter, uint32_t maxIterToDiversification, uint32_t *blocksStateCommunication, uint32_t *blocksBestEval,
		uint16_t *blocksBestSolution, uint32_t *hashTable)	{
	
	__shared__ uint32_t iter;
	__shared__ uint16_t iterBestI, iterBestJ;
	__shared__ uint32_t iterBestCost;

	__shared__ uint32_t blockBestCost;
	__shared__ uint16_t blockCurrentOrder[NUMBER_OF_ACTIVITIES];
	__shared__ uint16_t blockTabuIdx;
	__shared__ uint32_t *blockTabuList;
	__shared__ uint8_t *blockTabuCache;
	__shared__ bool blockReadGlobalBestSolution;
	__shared__ bool blockWriteGlobalBestSolution;
	__shared__ uint32_t blockNumberIterationSinceBest;
	__shared__ uint16_t blockResourceIdxs[NUMBER_OF_RESOURCES+1];


	uint16_t *threadStartTimesById = startTimesId+(blockIdx.x*blockDim.x+threadIdx.x)*numAct;

	if (threadIdx.x == 0)	{
		iter = 0;
		blockTabuIdx = 0;
		blockTabuList = tab+blockIdx.x*TABU_LIST_SIZE;
		blockTabuCache = tabCache+blockIdx.x*numAct*numAct;
		blockBestCost = bcost[blockIdx.x];
		for (uint8_t i = 0; i < NUMBER_OF_RESOURCES+1; ++i)	{
			blockResourceIdxs[i] = capIdxs[i];
		}
		blockNumberIterationSinceBest = 0;
		blockWriteGlobalBestSolution = false;
	}

	extern __shared__ uint64_t threadsResult[];
	#ifndef SHARED_ALLOC
	uint16_t threadResourcesLoad[TOTAL_SUM_OF_CAPACITY];
	uint16_t threadStartValues[MAXIMUM_CAPACITY_OF_RESOURCE];
	uint8_t threadReqItems[MAXIMUM_CAPACITY_OF_RESOURCE];
	#else
	uint16_t *basePtr = (uint16_t*) &threadsResult[blockDim.x];
	uint16_t *threadResourcesLoad = basePtr+threadIdx.x*TOTAL_SUM_OF_CAPACITY;
	uint16_t *threadStartValues = basePtr+blockDim.x*TOTAL_SUM_OF_CAPACITY+threadIdx.x*MAXIMUM_CAPACITY_OF_RESOURCE;
	uint8_t *threadReqItems = (uint8_t*) basePtr+blockDim.x*(TOTAL_SUM_OF_CAPACITY+MAXIMUM_CAPACITY_OF_RESOURCE);
	threadReqItems += threadIdx.x*MAXIMUM_CAPACITY_OF_RESOURCE;
	#endif

	for (uint16_t i = threadIdx.x; i < numAct; i += blockDim.x)	{
		blockCurrentOrder[i] = border[blockIdx.x*numAct+i];
	}

	__syncthreads();


	while (iter < maxIter)	{

		threadsResult[threadIdx.x] = 0xffffffffffffffff;

		for (uint32_t a = threadIdx.x+SWAP_RANGE; a < (numAct-1)*SWAP_RANGE; a += blockDim.x)	{
			uint16_t i = a/SWAP_RANGE;
			uint16_t j = (a % SWAP_RANGE)+i+1;
			if (j < numAct-1)	{
				uint32_t threadBestCost = threadsResult[threadIdx.x]>>32;
				uint32_t totalEval = cudaEvaluateOrder(numAct, numRes, blockCurrentOrder, i, j, threadStartTimesById, threadResourcesLoad, threadStartValues, threadReqItems, blockResourceIdxs, predIdx);
				totalEval += cudaComputePrecedencePenalty(numAct, sucIdx, threadStartTimesById);
			//	uint32_t hashTablePenalty = HASH_PENALTY_WEIGHT*hashTable[cudaComputeHashTableIndex(numAct, blockCurrentOrder, 0xffff, 0xffff, i, j)];
				uint32_t hashTablePenalty = HASH_PENALTY_WEIGHT*hashTable[cudaComputeHashTableIndex(numAct, blockCurrentOrder, i, j, 0, 0)];
		//		uint32_t hashTablePenalty = 0;
				bool isPossibleMove = cudaIsPossibleMove(numAct, i, j, blockTabuCache);
	//			bool isPossibleMove = true;

				if ((isPossibleMove && totalEval+hashTablePenalty < threadBestCost) || totalEval < blockBestCost)	{
/*					if (hashTablePenalty > 0)	{
						printf("Hash penalty: %d\n", hashTablePenalty);
					} */
					threadsResult[threadIdx.x] = 0ul;
					// Probability of lost best solution (if totalEval < blockBestCost) is small. 
					threadsResult[threadIdx.x] |= totalEval+hashTablePenalty;
					threadsResult[threadIdx.x] <<= 16;
					threadsResult[threadIdx.x] |= i;
					threadsResult[threadIdx.x] <<= 16;
					threadsResult[threadIdx.x] |= j;
				}
			}
		}
		__syncthreads();

		for (uint16_t k = blockDim.x/2; k > 0; k >>= 1)	{
			if (threadIdx.x < k)	{
				if ((threadsResult[threadIdx.x] & 0xffffffff00000000) > (threadsResult[threadIdx.x+k] & 0xffffffff00000000))
					threadsResult[threadIdx.x] = threadsResult[threadIdx.x+k];
			}
			__syncthreads();
		}

		if (threadIdx.x == 0)	{
			iterBestCost = threadsResult[0]>>32;
			iterBestJ = (threadsResult[0] & 0x000000000000ffff);
			iterBestI = (threadsResult[0] & 0x00000000ffff0000)>>16;
			if (atomicCAS(blocksStateCommunication, NOT_WRITED, WRITING_DATA) == NOT_WRITED || (*blocksBestEval >= iterBestCost && atomicCAS(blocksStateCommunication, DATA_AVAILABLE, WRITING_DATA) == DATA_AVAILABLE))	{
				blockWriteGlobalBestSolution = true;
			} else if ((blockNumberIterationSinceBest > maxIterToDiversification || blockBestCost > *blocksBestEval+10) && atomicCAS(blocksStateCommunication, DATA_AVAILABLE, READING_DATA) == DATA_AVAILABLE)	{
		//		printf("Read SOLUTION\n");
				blockReadGlobalBestSolution = true;	
				blockNumberIterationSinceBest = 0;
			}
		}
		
		if ((threadsResult[0] & 0x00000000ffffffff) == 0x00000000ffffffff)	{
			// Empty expanded neighborhood.
			break;
		}

		if (threadIdx.x == 0)	{
			// Aply best move.
			uint16_t t = blockCurrentOrder[iterBestI];
			blockCurrentOrder[iterBestI] = blockCurrentOrder[iterBestJ];
			blockCurrentOrder[iterBestJ] = t;
			// Add move to tabu list.
			cudaAddTurnToTabuList(numAct, iterBestI, iterBestJ, blockTabuList, blockTabuCache, blockTabuIdx);
		}
		__syncthreads();

		if (threadIdx.x == 1)	{
	//		atomicAdd(&hashTable[cudaComputeHashTableIndex(numAct, blockCurrentOrder, iterBestI, iterBestJ, iterBestI, iterBestJ)], 1);
			atomicAdd(&hashTable[cudaComputeHashTableIndex(numAct, blockCurrentOrder, 0xffff, 0xffff, 0, 0)], 1);
	//		__threadfence();
	//		printf("%d\n", cudaComputeHashTableIndex(numAct, blockCurrentOrder, iterBestI, iterBestJ, iterBestI, iterBestJ));
		}

		if (iterBestCost < blockBestCost)	{
			// Update best found solution.
			for (uint16_t i = threadIdx.x; i < numAct; i += blockDim.x)	{
				border[blockIdx.x*numAct+i] = blockCurrentOrder[i];
			}
			if (threadIdx.x == 0)	{
				blockBestCost = iterBestCost;
				blockNumberIterationSinceBest = 0;	
		//		printf("block id %d best cost: %d\n", blockIdx.x, blockBestCost);
			}
		} else if (threadIdx.x == 0 && !blockReadGlobalBestSolution) {
	//		printf("MISB: %d\n", blockNumberIterationSinceBest);
			++blockNumberIterationSinceBest;
		}

		if (blockWriteGlobalBestSolution == true)	{
			for (uint16_t i = threadIdx.x; i < numAct; i += blockDim.x)	{
				blocksBestSolution[i] = blockCurrentOrder[i];
			}
			if (threadIdx.x == 0)	{
				*blocksBestEval = iterBestCost;
		//		printf("write %d[%d]: %d\n",blockIdx.x, iter, *blocksBestEval);
			}
			__threadfence();
		}

		if (blockReadGlobalBestSolution == true)	{
			for (uint16_t i = threadIdx.x; i < numAct; i += blockDim.x)	{
				blockCurrentOrder[i] = blocksBestSolution[i];
			}
			// Clear tabu list and tabu cache.
			for (uint16_t i = threadIdx.x; i < TABU_LIST_SIZE; i += blockDim.x)	{
				uint32_t listElement = blockTabuList[i];
				uint16_t a = (listElement>>16);
				uint16_t b = (listElement & 0x0000ffff);
				blockTabuCache[a*numAct+b] = blockTabuCache[b*numAct+a] = 0;
				blockTabuList[i] = 0xffffffff;
			}

			if (threadIdx.x == 0)	{
				// No set blockBestCost variable!! This readed solution is recorded by other block.
				printf("read %d[%d]: %d\n",blockIdx.x,iter,*blocksBestEval);
			}
		}

		if (threadIdx.x == 0)	{
	//		if (iter % 100 == 0)
	//			printf("state: %d\n", *blocksStateCommunication);
			++iter;
		}
		__syncthreads();

		if (blockWriteGlobalBestSolution == true)	{
			atomicExch(blocksStateCommunication, DATA_AVAILABLE);
			blockWriteGlobalBestSolution = false;
		}

		if (blockReadGlobalBestSolution == true)	{
			atomicExch(blocksStateCommunication, DATA_AVAILABLE);
			blockReadGlobalBestSolution = false;
		}
	}

	if (threadIdx.x == 0)	{
		// Write best solution (with precedence + hash penalty). Cpu recompute penalty again.
		bcost[blockIdx.x] = blockBestCost;
	}
}


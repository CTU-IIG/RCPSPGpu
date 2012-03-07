#ifndef HLIDAC_PES_SCHEDULE_SOLVER_H
#define HLIDAC_PES_SCHEDULE_SOLVER_H

#include <iostream>
#include <stdint.h>
#include "CudaFunctions.cuh"

#define DEBUG_TABU_HASH 0

class ScheduleSolver {
	public:
		ScheduleSolver(uint8_t resNum, uint8_t *capRes, uint16_t actNum, uint8_t *actDur, uint16_t **actSuc, uint16_t *actNumSuc, uint8_t **actRes, bool verbose = false);

		void solveSchedule(const uint32_t& maxIter = 100, const uint32_t& maxIterToDiversification = 10);
		void printBestSchedule(bool verbose = true, std::ostream& OUT = std::cout) const;

		~ScheduleSolver();

	protected:

		void createInitialSolution(uint16_t *activitiesOrder);
		bool prepareCudaMemory(uint16_t *activitiesOrder, bool verbose);
		bool errorHandler(int16_t phase);
		uint16_t evaluateOrder(const uint16_t *order, uint16_t *startTimesWriter = NULL, uint16_t *startTimesWriterById = NULL) const;
		uint32_t computePrecedencePenalty(const uint16_t *startTimesById) const;
		void printSchedule(uint16_t *scheduleOrder, bool verbose = true, std::ostream& OUT = std::cout) const;
		bool checkSwapPrecedencePenalty(const uint16_t * const& order, const uint8_t * const& successorsMatrix, uint16_t i, uint16_t j) const;
		void makeDiversification(uint16_t * const& order, const uint8_t * const& successorsMatrix, const uint32_t numberOfSwaps);
		void freeCudaMemory();

	private:

		/* COPY OBJECT IS FORBIDDEN */

		ScheduleSolver(const ScheduleSolver&);
		ScheduleSolver& operator=(const ScheduleSolver&);


		/* IMMUTABLE DATA */

		// Number of renewable sources.
		uint8_t numberOfResources;
		// Capacity of resources;
		uint8_t *capacityOfResources;
		// Total number of activities.
		uint16_t numberOfActivities;
		// Duration of activities.
		uint8_t *activitiesDuration;
		// Activities successors;
		uint16_t **activitiesSuccessors;
		// Number of successors that activities.
		uint16_t *numberOfSuccessors;
		// Precomputed predecessors;
		uint16_t **activitiesPredecessors;
		// Number of predecessors;
		uint16_t *numberOfPredecessors;
		// Activities required sources.
		uint8_t **activitiesResources;
		//! Critical Path Makespan. (Critical Path Method)
		int32_t criticalPathMakespan;
		

		/* MUTABLE DATA */	

		// Best schedule order.
		bool solutionComputed;
		uint16_t *bestScheduleOrder;

		/* CUDA DATA */

		CudaData cudaData;
		uint16_t cudaCapability;
		uint16_t numberOfBlock;
		uint32_t dynSharedMemSize;
		uint32_t numberOfThreadsPerBlock;

		uint8_t *cudaActivitiesResourcesArray;
		uint16_t *cudaPredecessorsArray;	
		uint16_t *cudaPredecessorsIdxsArray;

		/* MISC DATA */

		// Total time of program running.
		double totalRunTime;
};

#endif


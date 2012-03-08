#ifndef HLIDAC_PES_SCHEDULE_SOLVER_CUH
#define HLIDAC_PES_SCHEDULE_SOLVER_CUH

/*!
 * \file ScheduleSolver.cuh
 * \author Libor Bukata
 * \brief RCPSP solver class.
 */

#include <iostream>
#include <stdint.h>
#include "CudaFunctions.cuh"
#include "InputReader.h"
#include "ConfigureRCPSP.h"

/*!
 * Constant that is used to turn on/off debug mode for GPU tabu hash.
 * If DEBUG is on then some tabu hash statistics are printed to console.
 */
#define DEBUG_TABU_HASH 0

/*!
 * Tabu search meta heuristic is used to solve RCPSP. GPU computing power is exploited for quicker solving of the problem.
 * \class ScheduleSolver
 * \brief Instance of this class is able to solve resource constrained project scheduling problem.
 */
class ScheduleSolver {
	public:
		/*!
		 * \param rcpspData Data of project instance.
		 * \param verbose If true then extra informations are printed.
		 * \exception runtime_error Cuda error occur.
		 * \brief Copy pointers of project data, initialize required structures, create initial activities order and copy data to GPU.
		 */
		ScheduleSolver(const InputReader& rcpspData, bool verbose = false);

		/*!
		 * \param maxIter Number of iterations that should be performed.
		 * \param maxIterSinceBest Maximal number of iterations since last improving move than another solution will be read.
		 * \exception runtime_error Cuda error occur.
		 * \brief Use GPU version of tabu search to find good quality solution.
		 */
		void solveSchedule(const uint32_t& maxIter = ConfigureRCPSP::NUMBER_OF_ITERATIONS, const uint32_t& maxIterSinceBest = ConfigureRCPSP::MAXIMAL_NUMBER_OF_ITERATIONS_SINCE_BEST);
		/*!
		 * \param verbose If false then only result makespan, critical path makespan and computational time will be printed.
		 * \param OUT Output stream.
		 * \brief Print best found schedule, schedule length and computational time.
		 */
		void printBestSchedule(bool verbose = true, std::ostream& OUT = std::cout) const;

		//! Free all allocated resources (CPU + GPU).
		~ScheduleSolver();

	protected:

		/*!
		 * \param activitiesOrder Initial activities order will be written to this array.
		 * \brief Compute predecessors and create initial activities order.
		 */
		void createInitialSolution(uint16_t *activitiesOrder);
		/*!
		 * \param activitiesOrder Sequence of the activities.
		 * \param verbose If true then more informations (Cuda info, etc.) will be showed.
		 * \return Return true if some Cuda error will be detected.
		 * \brief Copy required data to GPU and compute critical path makespan.
		 */
		bool prepareCudaMemory(uint16_t *activitiesOrder, bool verbose);
		/*!
		 * \param phase Number that correspond to a location at prepareCudaMemory method.
		 * \return Always return true.
		 * \brief Print error message, free Cuda allocated resources and return true.
		 */
		bool errorHandler(int16_t phase);
		/*!
		 * \param order Activities order.
		 * \param startTimesWriter Start times of the activities can be written to this array. Order is the same as at the order parameter.
		 * \param startTimesWriterById Start times of the activities can be written to this array. Order is defined by activities ID's.
		 * \return Length of the schedule.
		 * \brief Input order is evaluated and start times are determined. Total schedule length is returned.
		 */
		uint16_t evaluateOrder(const uint16_t * const& order, uint16_t *startTimesWriter = NULL, uint16_t *startTimesWriterById = NULL) const;
		/*!
		 * \param startTimesById Start times of activities ordered by ID's.
		 * \return Precedence penalty of the schedule.
		 * \brief Method compute precedence penalty (= broken relation between two activities) of the schedule.
		 * \note Because precedence free swaps and shifts are currently used, this function is only for debugging purposes.
		 */
		uint32_t computePrecedencePenalty(const uint16_t * const& startTimesById) const;
		/*!
		 * \param scheduleOrder Order of activities that should be evaluated.
		 * \param verbose If true then complete schedule will be printed else only basic information is printed.
		 * \param OUT Output stream.
		 * \brief Print schedule and schedule length. 
		 */
		void printSchedule(const uint16_t * const& scheduleOrder, bool verbose = true, std::ostream& OUT = std::cout) const;
		/*!
		 * \param order Order of activities.
		 * \param successorsMatrix Matrix of successors that is computed at prepareCudaMemory method.
		 * \param i Index at activitiesOrder.
		 * \param j Index at activitiesOrder.
		 * \return True if and only if precedence penalty is zero else false.
		 * \brief Method check if candidate for a swap is precedence penalty free.
		 */
		bool checkSwapPrecedencePenalty(const uint16_t * const& order, const uint8_t * const& successorsMatrix, uint16_t i, uint16_t j) const;
		/*!
		 * \param order Initial sequence of the activities.
		 * \param successorsMatrix Matrix of successors that is computed at prepareCudaMemory method.
		 * \param numberOfSwaps Number of successful (= zero precedence penalty) swaps.
		 * \brief Random swaps are performed at initial schedule.
		 */
		void makeDiversification(uint16_t * const& order, const uint8_t * const& successorsMatrix, const uint32_t& numberOfSwaps);
		//! Free all allocated Cuda resources.
		void freeCudaMemory();

	private:

		/* COPY OBJECT IS FORBIDDEN */

		//! Copy constructor is forbidden.
		ScheduleSolver(const ScheduleSolver&);
		//! Assignment operator is forbidden.
		ScheduleSolver& operator=(const ScheduleSolver&);


		/* IMMUTABLE DATA */

		//! Number of renewable sources.
		uint8_t numberOfResources;
		//! Capacity of resources;
		uint8_t *capacityOfResources;
		//! Total number of activities.
		uint16_t numberOfActivities;
		//! Duration of activities.
		uint8_t *activitiesDuration;
		//! Activities successors;
		uint16_t **activitiesSuccessors;
		//! Number of successors that activities.
		uint16_t *numberOfSuccessors;
		//! Precomputed predecessors.
		uint16_t **activitiesPredecessors;
		//! Number of predecessors.
		uint16_t *numberOfPredecessors;
		//! Sources that are required by activities.
		uint8_t **activitiesResources;
		//! Critical Path Makespan. (Critical Path Method)
		int32_t criticalPathMakespan;
		

		/* MUTABLE DATA */	

		//! If solution is successfully computed then solutionComputed variable is set to true.
		bool solutionComputed;
		//! Best schedule order.
		uint16_t *bestScheduleOrder;

		/* CUDA DATA */

		//! All required informations are passed through this variable to Cuda global function. For example pointers to device memory, integer parameters etc.
		CudaData cudaData;
		//! Cuda capability of selected graphics card. (for example value 130 correspond to capability 1.3)
		uint16_t cudaCapability;
		//! Number of blocks that should be launched on GPU.
		uint16_t numberOfBlock;
		//! Required amount of dynamic share memory allocation at GPU.
		uint32_t dynSharedMemSize;
		//! How many threads should be launched per one block.
		uint32_t numberOfThreadsPerBlock;

		//! Texture array of activities resource requirements.
		uint8_t *cudaActivitiesResourcesArray;
		//! Texture array of predecessors.
		uint16_t *cudaPredecessorsArray;
		//! Texture array of predecessors indices.
		uint16_t *cudaPredecessorsIdxsArray;

		/* MISC DATA */

		//! Purpose of this variable is to remember total computational time.
		double totalRunTime;
};

#endif


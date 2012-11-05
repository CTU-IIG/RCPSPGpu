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
		 * \param output Output stream.
		 * \brief Print best found schedule, schedule length and computational time.
		 */
		void printBestSchedule(bool verbose = true, std::ostream& output = std::cout) const;
		/*!
		 * \param fileName The name of the file where results will be written.
		 * \brief It writes required data structures and the best schedule to the given file.
		 */
		void writeBestScheduleToFile(const std::string& fileName);

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
		 * \param startActivityId The id of the start activity of the project.
		 * \param energyReasoning The energy requirements are taken into account if energyReasoning variable is set to true.
		 * \return The earliest start time for each activity.
		 * \brief Lower bounds of the earliest start time values are computed for each activity.
		 */
		uint16_t* computeLowerBounds(const uint16_t& startActivityId, const bool& energyReasoning = false) const;		

		/*!
		 * \param order Activities order.
		 * \param relatedActivities It's array of successors or predecessors. It depends on a way of evaluation (forward, backward).
		 * \param numberOfRelatedActivities Number of successors/predecessors for each activity.
		 * \param timeValuesById The earliest start time values for forward evaluation
		 * and transformed time values for backward evaluation.
		 * \param forwardEvaluation It determines if forward or backward schedule is evaluated.
		 * \return Length of the schedule.
		 * \brief Input order is evaluated and the earliest start/transformed time values are computed.
		 * \warning Order of activities is sequence of putting to the schedule, time values don't have to be ordered.
		 */
		uint16_t evaluateOrder(const uint16_t * const& order, const uint16_t * const * const& relatedActivities,
			       const uint16_t * const& numberOfRelatedActivities, uint16_t *& timeValuesById, bool forwardEvaluation) const;
		/*!
		 * \param order The sequence of putting to the schedule. It's activity order.
		 * \param startTimesById The earliest start time values for each scheduled activity.
		 * \return Project makespan, i.e. the length of the schedule.
		 * \brief It evaluates order of activities and determines the earliest start time values.
		 */
		uint16_t forwardScheduleEvaluation(const uint16_t * const& order, uint16_t *& startTimesById) const;
		/*!
		 * \param order The sequence of putting to the schedule. It's activity order.
		 * \param startTimesById The latest start time values for each scheduled activity.
		 * \return Project makespan, i.e. the length of the schedule.
		 * \brief It evaluates order (in reverse order) of activities and determines the latest start time values.
		 */
		uint16_t backwardScheduleEvaluation(const uint16_t * const& order, uint16_t *& startTimesById) const;
		/*!
		 * \param order Order of activities. It determines the order of putting to the schedule.
		 * \param bestScheduleStartTimesById The earliest start time values for the best found schedule.
		 * \return Project makespan, i.e. the length of the schedule.
		 * \brief Iterative method tries to shake down activities in the schedule to ensure equally loaded resources.
		 * Therefore, the shorter schedule could be found.
		 */
		uint16_t shakingDownEvaluation(const uint16_t * const& order, uint16_t *bestScheduleStartTimesById) const;
		/*!
		 * \param startTimesById Start times of activities ordered by ID's.
		 * \return Precedence penalty of the schedule.
		 * \brief Method compute precedence penalty (= broken relation between two activities) of the schedule.
		 * \note Because precedence free swaps and shifts are currently used, this function is only for debugging purposes.
		 */
		uint32_t computePrecedencePenalty(const uint16_t * const& startTimesById) const;
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
		 * \param order Original activities order.
		 * \param startTimesById The earliest start time values in the order W.
		 * \brief It transforms the earliest start time values to the order W. The order W is written to the variable order.
		 */
		void convertStartTimesById2ActivitiesOrder(uint16_t *order, const uint16_t * const& startTimesById) const;
		/*!
		 * \param order Order of activities.
		 * \param timeValuesById Assigned time values to activities, it is used for sorting input order.
		 * \param size It's size of the arrays, i.e. number of project activities.
		 * \brief Input order of activities is sorted in accordance with time values. It's stable sort.
		 */
		static void insertSort(uint16_t* order, const uint16_t * const& timeValuesById, const int32_t& size);
		
		/*!
		 * \param scheduleOrder Order of activities that should be evaluated.
		 * \param verbose If true then complete schedule will be printed else only basic information is printed.
		 * \param output Output stream.
		 * \brief Print schedule and schedule length. 
		 */
		void printSchedule(const uint16_t * const& scheduleOrder, bool verbose = true, std::ostream& output = std::cout) const;

		/*!
		 * \param order Initial sequence of the activities.
		 * \param successorsMatrix Matrix of successors that is computed at prepareCudaMemory method.
		 * \param numberOfSwaps Number of successful (= zero precedence penalty) swaps.
		 * \brief Random swaps are performed at initial schedule.
		 */
		void makeDiversification(uint16_t * const& order, const uint8_t * const& successorsMatrix, const uint32_t& numberOfSwaps);
		
		/*!
		 * \param activityId The activity from which all related activities are found.
		 * \param numberOfRelated The number of related activities for each activity.
		 * \param related The related (= successors || predecessors) activities for each activity in the project.
		 * \return It returns all activityId's successors or predecessors.
		 */
		std::vector<uint16_t> getAllRelatedActivities(uint16_t activityId, uint16_t *numberOfRelated, uint16_t **related) const;
		/*!
		 * \param activityId Identification of the activity.
		 * \return It returns all activityId's successors.
		 */
		std::vector<uint16_t> getAllActivitySuccessors(const uint16_t& activityId) const;
		/*!
		 * \param activityId Identification of the activity.
		 * \return It returns all activityId's predecessors.
		 */
		std::vector<uint16_t> getAllActivityPredecessors(const uint16_t& activityId) const;		

		/*!
		 * \param array The read-only array which will be copied.
		 * \param length The length of the array.
		 * \return The converted copy of the input array.
		 */
		template<class X, class Y>
		static Y* convertArrayType(X* array, size_t length);

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
		//! Upper bound of the project duration (sum of the all time flows).
		uint16_t upperBoundMakespan;
		

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
		//! Texture array of successors.
		uint16_t *cudaSuccessorsArray;
		//! Texture array of successors indices.
		uint16_t *cudaSuccessorsIdxsArray; 

		/* MISC DATA */

		//! Purpose of this variable is to remember total computational time.
		double totalRunTime;
		//! Number of evaluated schedules on the GPU.
		uint64_t numberOfEvaluatedSchedules;
};

#endif


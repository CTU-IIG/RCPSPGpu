#ifndef HLIDAC_PES_DEFAULT_CONFIGURE_RCPSP_H
#define HLIDAC_PES_DEFAULT_CONFIGURE_RCPSP_H

/* DEFAULT PARAMETERS FOR RCPSP - YOU CAN CHANGE. */

/*!
 * \file DefaultConfigureRCPSP.h
 * \author Libor Bukata
 * \brief Default settings for RCPSPGpu.
 */


/* SCHEDULE SOLVER */

//! Tabu list size. 
#define DEFAULT_TABU_LIST_SIZE 800	// Good choice: 120 activities - 800; 30 activities - 60
//! If you want to use tabu hash then set this value to 1. 
#define DEFAULT_USE_TABU_HASH 0
//! Tabu search iterations per block.
#define DEFAULT_NUMBER_OF_ITERATIONS 500 
//! Maximal number of iterations since best solution found. (diversification purposes)
#define DEFAULT_MAXIMAL_NUMBER_OF_ITERATIONS_SINCE_BEST 300
//! Define maximal distance of swapped activities.
#define DEFAULT_SWAP_RANGE 60
//! Number of diversification swaps.
#define DEFAULT_DIVERSIFICATION_SWAPS 10
//! Total number of generated set solutions.
#define DEFAULT_NUMBER_OF_SET_SOLUTIONS 8
//! Number of blocks (independent search processes) per multiprocessor of GPU.
#define DEFAULT_NUMBER_OF_BLOCKS_PER_MULTIPROCESSOR 2
//! Maximal value of read counter then diversification will be called.
#define DEFAULT_MAXIMAL_VALUE_OF_READ_COUNTER 3

#endif


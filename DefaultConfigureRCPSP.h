/*
	This file is part of the RCPSPGpu program.

	RCPSPGpu is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	RCPSPGpu is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with RCPSPGpu. If not, see <http://www.gnu.org/licenses/>.
*/
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
//! Tabu search iterations per block.
#define DEFAULT_NUMBER_OF_ITERATIONS 500 
//! Define maximal distance of swapped activities.
#define DEFAULT_SWAP_RANGE 60
//! Number of diversification swaps.
#define DEFAULT_DIVERSIFICATION_SWAPS 20
//! Total number of generated set solutions.
#define DEFAULT_NUMBER_OF_SET_SOLUTIONS 8
//! Number of blocks (independent search processes) per multiprocessor of GPU.
#define DEFAULT_NUMBER_OF_BLOCKS_PER_MULTIPROCESSOR 2
//! Maximal value of read counter then diversification will be called.
#define DEFAULT_MAXIMAL_VALUE_OF_READ_COUNTER 3
//! If you want to write the best schedule to a file set this variable to 1.
#define DEFAULT_WRITE_RESULT_FILE 0

#endif


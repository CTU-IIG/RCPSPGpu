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
#ifndef HLIDAC_PES_CONFIGURE_RCPSP_H
#define HLIDAC_PES_CONFIGURE_RCPSP_H

/*!
 * \file ConfigureRCPSP.h
 * \author Libor Bukata
 * \brief Represents a setting of RCPSP.
 */

#include <stdint.h>
#include "DefaultConfigureRCPSP.h"

/*!
 * \namespace ConfigureRCPSP
 * \brief Configurable extern global variables are defined at this namespace.
 */
namespace ConfigureRCPSP {

	/* SCHEDULE SOLVER SETTINGS */

	//! Tabu list size.
	extern uint32_t TABU_LIST_SIZE;
	//! Number of search iterations.
	extern uint32_t NUMBER_OF_ITERATIONS;
	//! Maximal distance between swapped activities.
	extern uint32_t SWAP_RANGE;
	//! Number of diversification swaps.
	extern uint32_t DIVERSIFICATION_SWAPS;
	//! Number of solutions at working set.
	extern uint32_t NUMBER_OF_SET_SOLUTIONS;
	//! Number of blocks per multiprocessor.
	extern uint32_t NUMBER_OF_BLOCKS_PER_MULTIPROCESSOR;
	//! Maximal value of read counter for a set solution.
	extern uint32_t MAXIMAL_VALUE_OF_READ_COUNTER;
	//! Do you want to write a result file with the encoded best schedule?
	extern bool WRITE_RESULT_FILE;
}

#endif


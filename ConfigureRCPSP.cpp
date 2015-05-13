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
#include <stdint.h>
#include "ConfigureRCPSP.h"

namespace ConfigureRCPSP {
	/* SCHEDULE SOLVER SETTINGS */
	uint32_t TABU_LIST_SIZE = DEFAULT_TABU_LIST_SIZE;
	uint32_t NUMBER_OF_ITERATIONS = DEFAULT_NUMBER_OF_ITERATIONS;
	uint32_t SWAP_RANGE = DEFAULT_SWAP_RANGE;
	uint32_t DIVERSIFICATION_SWAPS = DEFAULT_DIVERSIFICATION_SWAPS;
	uint32_t NUMBER_OF_SET_SOLUTIONS = DEFAULT_NUMBER_OF_SET_SOLUTIONS;
	uint32_t NUMBER_OF_BLOCKS_PER_MULTIPROCESSOR = DEFAULT_NUMBER_OF_BLOCKS_PER_MULTIPROCESSOR;
	uint32_t MAXIMAL_VALUE_OF_READ_COUNTER = DEFAULT_MAXIMAL_VALUE_OF_READ_COUNTER;
	bool WRITE_RESULT_FILE = (DEFAULT_WRITE_RESULT_FILE == 1 ? true : false);
}


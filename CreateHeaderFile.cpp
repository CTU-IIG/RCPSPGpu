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
/*!
 * \file CreateHeaderFile.cpp
 * \author Libor Bukata
 * \brief Program that analyse, compute and create CudaConstants.h file.
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "InputReader.h"

using namespace std;

// Forward declaration entry point of the program.
int createHeaderFile(int argc, char* argv[]);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// Trick how can be entry point of the program ,,renamed'' from main to createHeaderFile. (for purposes of documentation - Doxygen)
int main(int argc, char* argv[])	{
	return createHeaderFile(argc,argv);
}
#endif

/*!
 * \param argc Number of command line arguments.
 * \param argv Command line arguments.
 * \return Zero if success else error code (positive number).
 * \brief Process instance files, compute statistics, write Cuda header file...
 */
int createHeaderFile(int argc, char* argv[])	{
	if (argc < 2)	{
		cerr<<"Invalid number of parameters!"<<endl;
		cerr<<"Use: ./CreateHeaderFile instance_file1 instance_file2 ..."<<endl;
		return 1;
	}

	vector<string> filenames;
	for (int i = 1; i < argc; ++i)	{
		filenames.push_back(argv[i]);
	}

	try {
		writeHeaderFile(filenames, "CudaConstants.h");
	} catch (exception& e)	{
		cerr<<e.what()<<endl;
		return 2;
	}

	return 0;
}


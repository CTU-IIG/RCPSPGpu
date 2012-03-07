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


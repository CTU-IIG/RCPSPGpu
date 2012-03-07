/*!
 * \file RCPSPGpu.cpp
 * \author Libor Bukata
 * \brief RCPSP - Resource Constrained Project Scheduling Problem - GPU version.
 */

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "InputReader.h"
#include "ConfigureRCPSP.h"
#include "ScheduleSolver.cuh"

using namespace std;

// Forward declaration entry point of the program.
int rcpspGpu(int argc, char* argv[]);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// Trick how can be entry point of the program ,,renamed'' from main to rcpspGpu. (for purposes of documentation - Doxygen)
int main(int argc, char* argv[])	{
	return rcpspGpu(argc,argv);
}
#endif

/*!
 * \param option Command line switch.
 * \param i Current index at argv two-dimensional array. Can be modified.
 * \param argc Number of arguments (program name + switches + parameters) that were given through command line.
 * \param argv Command line arguments.
 * \exception invalid_argument Parameter cannot be read.
 * \exception range_error Invalid value of read parameter.
 * \return Parameter of the switch.
 * \brief Helper function for command line processing.
 */
int optionHelper(const string& option, int& i, const int& argc, char* argv[])	{
	if (i+1 < argc)	{
		int value = 0;
		string numStr = argv[++i];
		istringstream istr(numStr, istringstream::in);
		if (!(istr>>value))
			throw invalid_argument("Cannot read parameter! (option \""+option+"\")");
		if (value <= 0 || (!numStr.empty() && numStr[0] == '-'))
			throw range_error("Parameter value cannot be negative or zero!");
		return value;
	} else {
		throw invalid_argument("Option \""+option+"\" require argument!");
	}
}

/*!
 * Entry point for RCPSP solver. Command line arguments are processed, input instances are
 * read and solved. Results are printed to console (can be easily redirected to file). 
 * Verbose mode is turned on if and only if one input file is read.
 * \param argc Number of command line arguments.
 * \param argv Command line arguments.
 * \return Zero if success else error code (positive number).
 * \brief Process arguments, read instances, solve instances and print results.
 */
int rcpspGpu(int argc, char* argv[])	{

	vector<string> inputFiles;

	for (int i = 1; i < argc; ++i)	{

		string arg = argv[i];

		if (arg == "--input-files" || arg == "-if")	{
			if (i+1 < argc)	{
				while (i+1 < argc && argv[i+1][0] != '-')	{
					inputFiles.push_back(argv[++i]);
				}
			} else {
				cerr<<"Option \"--input-files\" require parameter(s)!"<<endl;
				return 1;
			}
		}

		try {
			if (arg == "--number-of-iterations-per-block" || arg == "-noipb" || arg == "-noi")
				ConfigureRCPSP::NUMBER_OF_ITERATIONS = optionHelper("--number-of-iterations-per-block", i, argc, argv);
			if (arg == "--max-iter-since-best" || arg == "-misb")
				ConfigureRCPSP::MAXIMAL_NUMBER_OF_ITERATIONS_SINCE_BEST = optionHelper("--max-iter-since-best", i, argc, argv);
			if (arg == "--tabu-list-size" || arg == "-tls")
				ConfigureRCPSP::TABU_LIST_SIZE = optionHelper("--tabu-list-size", i, argc, argv);
			if (arg == "--swap-range" || arg == "-sr" || arg == "-swr")
				ConfigureRCPSP::SWAP_RANGE = optionHelper("--swap-range", i, argc, argv);
			if (arg == "--diversification-swaps" || arg == "-ds")
				ConfigureRCPSP::DIVERSIFICATION_SWAPS = optionHelper("--diversification-swaps", i, argc, argv);
			if (arg == "--number-of-set-solutions" || arg == "-noss")
				ConfigureRCPSP::NUMBER_OF_SET_SOLUTIONS = optionHelper("--number-of-set-solutions", i, argc, argv);
			if (arg == "--number-of-blocks-per-multiprocessor" || arg == "-nobpm")
				ConfigureRCPSP::NUMBER_OF_BLOCKS_PER_MULTIPROCESSOR = optionHelper("--number-of-blocks-per-multiprocessor", i, argc, argv);
			if (arg == "--maximal-value-of-read-counter" || arg == "-mvorc")
				ConfigureRCPSP::MAXIMAL_VALUE_OF_READ_COUNTER = optionHelper("--maximal-value-of-read-counter", i, argc, argv);
		} catch (exception& e)	{
			cerr<<e.what()<<endl;
			return 1;
		}

		if (arg == "--use-tabu-hash" || arg == "-uth")
			ConfigureRCPSP::USE_TABU_HASH = true;

		if (arg == "--help" || arg == "-h")	{
			cout<<"RCPSP schedule solver."<<endl<<endl;
			cout<<"Usage:"<<endl;
			cout<<"\t"<<argv[0]<<" [options+parameters] --input-files file1 file2 ..."<<endl;
			cout<<"Options:"<<endl;
			cout<<"\t"<<"--input-files ARG, -if ARG, ARG=\"FILE1 FILE2 ... FILEX\""<<endl;
			cout<<"\t\t"<<"Instances data. Input files are delimited by space."<<endl;
			cout<<"\t"<<"--number-of-iterations-per-block ARG, -noipb ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"Number of iterations per block after which search process will be stopped."<<endl;
			cout<<"\t"<<"--max-iter-since-best ARG, -misb ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"Maximal number of iterations without improving solution after which another solution will be read."<<endl;
			cout<<"\t"<<"--tabu-list-size ARG, -tls ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"Size of tabu list located at GPU memory."<<endl;
			cout<<"\t"<<"--swap-range ARG, -sr ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"Maximal distance between swapped activities."<<endl;
			cout<<"\t"<<"--diversification-swaps ARG, -ds ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"How many swaps should be performed than diversification is callled."<<endl;
			cout<<"\t"<<"--number-of-set-solutions ARG, -noss ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"How many set solutions should be created."<<endl;
			cout<<"\t"<<"--number-of-blocks-per-multiprocessor ARG, -nobpm ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"Number of Cuda blocks that should be launched per one multiprocessor."<<endl;
			cout<<"\t"<<"--use-tabu-hash, -uth"<<endl;
			cout<<"\t\t"<<"If you add this switch then tabu hash will be used. That could improve quality of solution."<<endl;
			cout<<"\t"<<"--maximal-value-of-read-counter ARG, -mvorc ARG, ARG=POSITIVE_INTEGER"<<endl;
			cout<<"\t\t"<<"How many times will be new (updated) set solution read without diversification."<<endl;
			cout<<"\t\t"<<"If set solution is changed - improved, then counter is reset."<<endl;
			cout<<"\t\t"<<"Every Cuda thread performs diversification of read solution if and only if counter value is greater than ARG."<<endl<<endl;
			cout<<"Default values can be modified at \"DefaultConfigureRCPSP.h\" file."<<endl;
			return 0;
		}
	}


	try {
		bool verbose = (inputFiles.size() == 1 ? true : false);
		for (vector<string>::const_iterator it = inputFiles.begin(); it != inputFiles.end(); ++it)	{
			// Filename of instance.
			string filename = *it;
			InputReader reader;
			// Read instance data.
			reader.readFromFile(filename);
			// Init schedule solver.
			ScheduleSolver solver(reader, verbose);
			// Solve read instance.	
			solver.solveSchedule(ConfigureRCPSP::NUMBER_OF_ITERATIONS, ConfigureRCPSP::MAXIMAL_NUMBER_OF_ITERATIONS_SINCE_BEST);
			// Print results.
			if (verbose == true)	{
				solver.printBestSchedule();
			}	else	{
				cout<<filename<<": ";
				solver.printBestSchedule(false);
			}
		}
	} catch (exception& e)	{
		cerr<<e.what()<<endl;
		return 2;
	}

	return 0;
}


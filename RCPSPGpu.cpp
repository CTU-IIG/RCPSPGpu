#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "InputReader.h"
#include "ScheduleSolver.cuh"

using namespace std;

int main(int argc, char* argv[])	{

	vector<string> inputFiles;
	uint32_t maxIterSinceBest = 100;
	uint32_t numberOfIteration = 1000;

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

		if (arg == "--number-of-iter" || arg == "-nit")	{
			if (i+1 < argc)	{
				string numStr = argv[++i];				
				istringstream istr(numStr,istringstream::in);
				istr>>numberOfIteration;
			} else {
				cerr<<"Option \"--number-of-iter\" require parameter!"<<endl;
				return 1;
			}
		}

		if (arg == "--max-iter-since-best" || arg == "-misb")	{
			if (i+1 < argc)	{
				string iterMax = argv[++i];				
				istringstream istr(iterMax,istringstream::in);
				istr>>maxIterSinceBest;
			} else {
				cerr<<"Option \"--max-iter-since-best\" require parameter!"<<endl;
				return 1;
			}
		}

		if (arg == "--help" || arg == "-h")	{
			cout<<"TODO: Print help..."<<endl;
			return 0;
		}
	}


	try {
		bool verbose = (inputFiles.size() == 1 ? true : false);
		for (vector<string>::const_iterator it = inputFiles.begin(); it != inputFiles.end(); ++it)	{
			string filename = *it;
			InputReader reader;
			reader.readFromFile(filename);

			ScheduleSolver solver(reader, verbose);
			solver.solveSchedule(numberOfIteration, maxIterSinceBest);

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


#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "InputReader.h"

using namespace std;

int main(int argc, char* argv[])	{
	if (argc < 2)	{
		cerr<<"Invalid number of parameters!"<<endl;
		cerr<<"Use: ./CreateHeaderFile filename1 filename2 ..."<<endl;
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


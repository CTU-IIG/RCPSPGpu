#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include "InputReader.h"

using namespace std;

void InputReader::readFromFile(const string& filename) {
	ifstream IN(filename.c_str());	
	if (!IN)
		throw runtime_error("InputReader::readFromFIle: Cannot open input file!");

	readFromStream(IN);
	IN.close();
}

void InputReader::readFromStream(istream& IN) {

	freeInstanceData();

	uint16_t shred;
	string readedLine;

	string::const_iterator sit;
	string searchPattern1 = "- renewable";
	string searchPattern2 = "MPM-Time";
	string searchPattern3 = "#successors";
	
	// Read project basic parameters.
	while (getline(IN,readedLine))	{
		if ((sit = search(readedLine.begin(), readedLine.end(), searchPattern1.begin(), searchPattern1.end())) != readedLine.end())	{
			string parsedNumber;
			for (string::const_iterator it = sit; it != readedLine.end(); ++it)	{
				if (isdigit(*it) > 0)
					parsedNumber.push_back(*it);			
			}
			totalNumberOfResources = strToNumber(parsedNumber);
		}

		if ((sit = search(readedLine.begin(), readedLine.end(), searchPattern2.begin(), searchPattern2.end())) != readedLine.end())	{
			IN>>shred>>numberOfActivities;
			numberOfActivities += 2;
		}

		if ((sit = search(readedLine.begin(), readedLine.end(), searchPattern3.begin(), searchPattern3.end())) != readedLine.end())	{
			break;
		}
	}

	activitiesDuration = new uint8_t[numberOfActivities];
	activitiesRequiredResources = new uint8_t*[numberOfActivities];
	activitiesSuccessors = new uint16_t*[numberOfActivities];
	activitiesNumberOfSuccessors = new uint16_t[numberOfActivities];
	capacityOfResources = new uint8_t[totalNumberOfResources];
	
	// Read succesors.
	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		size_t numberOfSuccessors;
		IN>>shred>>shred>>numberOfSuccessors;

		activitiesNumberOfSuccessors[activityId] = numberOfSuccessors;
		activitiesSuccessors[activityId] = new uint16_t[numberOfSuccessors];

		uint16_t successor;
		for (uint16_t i = 0; i < numberOfSuccessors; ++i)	{
			IN>>successor;
			activitiesSuccessors[activityId][i] = successor-1;
		}
	}

	for (uint8_t i = 0; i < 5; ++i)
		getline(IN,readedLine);

	// Read resource requirements.
	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		uint16_t activityDuration;
		IN>>shred>>shred>>activityDuration;

		activitiesDuration[activityId] = (uint16_t) activityDuration;
		activitiesRequiredResources[activityId] = new uint8_t[totalNumberOfResources];

		for (uint8_t resourceId = 0; resourceId < totalNumberOfResources; ++resourceId)	{
			uint16_t unitsReq; IN>>unitsReq;
			activitiesRequiredResources[activityId][resourceId] = (uint8_t) unitsReq;
		}
	}

	for (uint8_t i = 0; i < 4; ++i)
		getline(IN,readedLine);

	// Read capacity of resources.
	for (uint8_t resourceId = 0; resourceId < totalNumberOfResources; ++resourceId)	{
		uint16_t resourceCapacity; IN>>resourceCapacity;
		capacityOfResources[resourceId] = (uint8_t) resourceCapacity;
	}
}

void InputReader::printInstance(ostream& OUT)	const	{
	for (uint16_t actId = 0; actId < numberOfActivities; ++actId)	{
		OUT<<string(50,'+')<<endl;
		OUT<<"Activity number: "<<actId+1<<endl;
		OUT<<"Duration of activity: "<<(uint16_t) activitiesDuration[actId]<<endl;	
		OUT<<"Required sources (Resource ID : Units required):"<<endl;
		for (uint8_t *resPtr = activitiesRequiredResources[actId]; resPtr < activitiesRequiredResources[actId]+totalNumberOfResources; ++resPtr)	{
			OUT<<"\t("<<((resPtr-activitiesRequiredResources[actId])+1)<<" : "<<(uint16_t) *resPtr<<")"<<endl;
		}
		OUT<<"Successors of activity:";
		for (uint16_t *sucPtr = activitiesSuccessors[actId]; sucPtr < activitiesSuccessors[actId]+activitiesNumberOfSuccessors[actId]; ++sucPtr)	{
			OUT<<" "<<*sucPtr+1;
		}
		OUT<<endl;
		OUT<<string(50,'-')<<endl;
	}
	OUT<<"Max capacity of resources:";
	for (uint8_t *capPtr = capacityOfResources; capPtr < capacityOfResources+totalNumberOfResources; ++capPtr)
		OUT<<" "<<(uint16_t) *capPtr;
	OUT<<endl;
}


uint16_t InputReader::strToNumber(const string& number)	const {
	istringstream istr(number,istringstream::in);
	uint16_t ret; istr>>ret;
	return ret;
}

void InputReader::freeInstanceData()	{
	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		delete[] activitiesSuccessors[activityId];
		delete[] activitiesRequiredResources[activityId];
	}

	delete[] activitiesSuccessors;	
	delete[] activitiesRequiredResources;
	delete[] activitiesDuration;
	delete[] activitiesNumberOfSuccessors;
	delete[] capacityOfResources;
}


void writeHeaderFile(const vector<string>& inputFiles, const string& headerFile)	{

	ofstream OUT(headerFile.c_str());
	if (!OUT)
		throw runtime_error("writeHeaderFile: Cannot write to header file! Check permision!");

	uint16_t maxNumberOfActivities = 0, maxNumberOfResources = 0, maxCapacityOfResource = 0, maxTotalCapacity = 0;

	for (vector<string>::const_iterator it = inputFiles.begin(); it != inputFiles.end(); ++it)	{
		InputReader reader;
		reader.readFromFile(*it);

		maxNumberOfActivities = max(maxNumberOfActivities, reader.numberOfActivities);
		maxNumberOfResources = max(maxNumberOfResources, (uint16_t) reader.totalNumberOfResources);
		maxCapacityOfResource = max(maxCapacityOfResource, (uint16_t) *max_element(reader.capacityOfResources, reader.capacityOfResources+reader.totalNumberOfResources));

		uint16_t totalCapacity = 0;
		for (uint8_t *ptr = reader.capacityOfResources; ptr < reader.capacityOfResources+reader.totalNumberOfResources; ++ptr)	{
			totalCapacity += *ptr;
		}
		maxTotalCapacity = max(maxTotalCapacity, totalCapacity);
	}

	OUT<<"#ifndef HLIDAC_PES_CUDA_CONSTANTS_H"<<endl;
	OUT<<"#define HLIDAC_PES_CUDA_CONSTANTS_H"<<endl;
	OUT<<endl;
	OUT<<"#define NUMBER_OF_ACTIVITIES "<<maxNumberOfActivities<<endl;
	OUT<<"#define NUMBER_OF_RESOURCES "<<maxNumberOfResources<<endl;
	OUT<<"#define MAXIMUM_CAPACITY_OF_RESOURCE "<<maxCapacityOfResource<<endl;
	OUT<<"#define TOTAL_SUM_OF_CAPACITY "<<maxTotalCapacity<<endl;
	OUT<<endl;
	OUT<<"/* Hash constants. */"<<endl;
	OUT<<"#define HASH_TABLE_SIZE 16777216"<<endl;
	OUT<<"#define R 3144134277"<<endl;
	OUT<<endl;
	OUT<<"/* Blocks communication. */"<<endl;
	OUT<<"#define DATA_AVAILABLE 0"<<endl;
	OUT<<"#define DATA_ACCESS 1"<<endl;
	OUT<<endl;
	OUT<<"#endif"<<endl<<endl;

	OUT.close();
}



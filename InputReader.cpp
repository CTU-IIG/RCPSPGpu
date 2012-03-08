#include <algorithm>
#include <cctype>
#include <climits>
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
		throw runtime_error("InputReader::readFromFile: Cannot open input file \""+filename+"\"!");

	readFromStream(IN);
	IN.close();
}

void InputReader::readFromStream(istream& IN) {

	freeInstanceData();

	uint32_t shred;
	string readLine;

	if (getline(IN, readLine))	{
		IN.seekg (0, ios::beg);
		
		if (!readLine.empty() && readLine[0] == '*')	{
			/* PROGEN-SFX FORMAT */
			string::const_iterator sit;
			string searchPattern1 = "- renewable";
			string searchPattern2 = "MPM-Time";
			string searchPattern3 = "#successors";

			// Read project basic parameters.
			while (getline(IN,readLine))	{
				if ((sit = search(readLine.begin(), readLine.end(), searchPattern1.begin(), searchPattern1.end())) != readLine.end())	{
					string parsedNumber;
					for (string::const_iterator it = sit; it != readLine.end(); ++it)	{
						if (isdigit(*it) > 0)
							parsedNumber.push_back(*it);			
					}
					uint32_t numberOfResources = strToNumber(parsedNumber);
					if (parsedNumber.empty() || numberOfResources == 0)
						throw runtime_error("InputReader::readFromStream: Cannot read number of resources!");
					if (numberOfResources > UCHAR_MAX)
						throw runtime_error("InputReader::readFromStream: Maximal number of resources is "+numberToStr(UCHAR_MAX)+"!");
					totalNumberOfResources = numberOfResources;
				}

				if ((sit = search(readLine.begin(), readLine.end(), searchPattern2.begin(), searchPattern2.end())) != readLine.end())	{
					uint32_t readNumberOfActivities = 0;
					if (!(IN>>shred>>readNumberOfActivities))
						throw runtime_error("InputReader::readFromStream: Cannot read number of activities!");
					if (readNumberOfActivities == 0)
						throw runtime_error("InputReader::readFromStream: Number of activities is number greater than zero!");
					readNumberOfActivities += 2;
					if (readNumberOfActivities > USHRT_MAX)
						throw runtime_error("InputReader::readFromStream: Maximal number of activities is "+numberToStr(USHRT_MAX)+"!");
					numberOfActivities = readNumberOfActivities;
				}

				if ((sit = search(readLine.begin(), readLine.end(), searchPattern3.begin(), searchPattern3.end())) != readLine.end())	{
					break;
				}
			}

			if (numberOfActivities == 0 || totalNumberOfResources == 0)
				throw runtime_error("InputReader::readFromStream: Invalid format of input file!");

			allocateBaseArrays();

			// Read succesors.
			for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
				uint32_t numberOfSuccessors, testId;
				if (!(IN>>testId>>shred>>numberOfSuccessors))
					throw runtime_error("InputReader::readFromStream: Cannot read successors of activity "+numberToStr(activityId+1)+"!");
				if (((uint32_t) activityId+1) != testId)
					throw runtime_error("InputReader::readFromStream: Probably inconsistence of instance file!\nCheck activity ID failed.");
				if (numberOfSuccessors > USHRT_MAX)
					throw runtime_error("InputReader::readFromStream: Maximal number of successors per one activity is "+numberToStr(USHRT_MAX)+"!");

				activitiesNumberOfSuccessors[activityId] = (uint16_t) numberOfSuccessors;
				activitiesSuccessors[activityId] = new uint16_t[numberOfSuccessors];

				uint32_t successor;
				for (uint16_t i = 0; i < numberOfSuccessors; ++i)	{
					if (!(IN>>successor))
						throw runtime_error("InputReader::readFromStream: Cannot read next ("+numberToStr(i+1)+") successor of activity "+numberToStr(activityId+1)+"!");
					if (successor > numberOfActivities)
						throw runtime_error("InputReader::readFromStream: Invalid successor ID of activity "+numberToStr(activityId+1)+"!");
					activitiesSuccessors[activityId][i] = (uint16_t) successor-1;
				}
			}

			for (uint32_t i = 0; i < 5; ++i)
				getline(IN,readLine);

			// Read resource requirements.
			for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
				uint32_t activityDuration, testId;
				if (!(IN>>testId>>shred>>activityDuration))
					throw runtime_error("InputReader::readFromStream: Invalid read of activity requirement!");
				if (((uint32_t) activityId+1) != testId)
					throw runtime_error("InputReader::readFromStream: Probably inconsistence of instance file!\nCheck activity ID failed.");
				if (activityDuration > UCHAR_MAX)
					throw runtime_error("InputReader::readFromStream: Maximal duration of an activity is "+numberToStr(UCHAR_MAX)+"!");

				activitiesDuration[activityId] = (uint8_t) activityDuration;

				uint32_t unitsReq; 
				for (uint8_t resourceId = 0; resourceId < totalNumberOfResources; ++resourceId)	{
					if (!(IN>>unitsReq))
						throw runtime_error("InputReader::readFromStream: Cannot read next activity requirement ("+numberToStr(resourceId+1)+") of activity "+numberToStr(activityId+1)+"!");
					if (unitsReq > UCHAR_MAX)
						throw runtime_error("InputReader::readFromStream: Maximal activity resource requirement is "+numberToStr(UCHAR_MAX)+"!");

					activitiesRequiredResources[activityId][resourceId] = (uint8_t) unitsReq;
				}
			}

			for (uint8_t i = 0; i < 4; ++i)
				getline(IN,readLine);

			// Read capacity of resources.
			for (uint8_t resourceId = 0; resourceId < totalNumberOfResources; ++resourceId)	{
				uint32_t resourceCapacity;
				if (!(IN>>resourceCapacity))
					throw runtime_error("InputReader::readFromStream: Invalid read of resource capacity!\nResource ID is "+numberToStr(resourceId+1)+".");
				if (resourceCapacity > UCHAR_MAX)
					throw runtime_error("InputReader::readFromStream: Maximal capacity of resource is "+numberToStr(UCHAR_MAX)+"!");

				capacityOfResources[resourceId] = (uint8_t) resourceCapacity;
			}
		} else if (!readLine.empty())	{
			/* PROGEN/MAX 1.0 FORMAT */
			uint32_t readNumberOfActivities = 0, readNumberOfResources = 0;
			if (!(IN>>readNumberOfActivities>>readNumberOfResources>>shred>>shred))
				throw runtime_error("InputReader::readFromStream: Cannot read number of activities and number of resource!\nCheck file format.");
			if (readNumberOfActivities == 0 || readNumberOfResources == 0)
				throw runtime_error("InputReader::readFromStream: Invalid value of number of activities or number of resources!");
			readNumberOfActivities += 2;
			if (readNumberOfActivities > USHRT_MAX || readNumberOfResources > UCHAR_MAX)
				throw runtime_error("InputReader::readFromStream: Maximal number of resources is "+numberToStr(UCHAR_MAX)+" and maximal number of activities is "+numberToStr(USHRT_MAX)+"!");

			numberOfActivities = readNumberOfActivities;
			totalNumberOfResources = readNumberOfResources;

			allocateBaseArrays();

			// Read activity successors.
			for (uint16_t actId = 0; actId < numberOfActivities; ++actId)	{
				uint32_t successor = 0, numberOfSuccessors = 0, testId = 0;
				if (!(IN>>testId>>shred>>numberOfSuccessors))
					throw runtime_error("InputReader::readFromStream: Cannot read number of successors of activity "+numberToStr(actId)+"!");
				if (actId != testId)
					throw runtime_error("InputReader::readFromStream: Probably inconsistence of instance file!\nCheck activity ID failed.");
				if (numberOfSuccessors > USHRT_MAX)
					throw runtime_error("InputReader::readFromStream: Maximal number of successors per one activity is "+numberToStr(USHRT_MAX)+"!");

				activitiesNumberOfSuccessors[actId] = (uint16_t) numberOfSuccessors;
				activitiesSuccessors[actId] = new uint16_t[numberOfSuccessors];
				for (uint16_t k = 0; k < numberOfSuccessors; ++k)	{
					if (!(IN>>successor))
						throw runtime_error("InputReader::readFromStream: Cannot read next ("+numberToStr(k+1)+") successor of activity "+numberToStr(actId)+"!");
					if (successor >= numberOfActivities)
						throw runtime_error("InputReader::readFromStream: Invalid successor ID of activity "+numberToStr(actId)+"!");
					activitiesSuccessors[actId][k] = (uint16_t) successor;
				}
			}

			// Read activities resources requirement.
			for (uint16_t actId = 0; actId < numberOfActivities; ++actId)	{
				uint32_t resourceReq = 0, activityDuration = 0, testId = 0;
				if (!(IN>>testId>>shred>>activityDuration))
					throw runtime_error("InputReader::readFromStream: Invalid read of activity requirement!");
				if (actId != testId)
					throw runtime_error("InputReader::readFromStream: Probably inconsistence of instance file!\nCheck activity ID failed.");
				if (activityDuration > UCHAR_MAX)
					throw runtime_error("InputReader::readFromStream: Maximal duration of an activity is "+numberToStr(UCHAR_MAX)+"!");

				activitiesDuration[actId] = (uint8_t) activityDuration;

				for (uint8_t r = 0; r < totalNumberOfResources; ++r)	{
					if (!(IN>>resourceReq))
						throw runtime_error("InputReader::readFromStream: Cannot read next activity requirement ("+numberToStr(r)+") of activity "+numberToStr(actId)+"!");
					if (resourceReq > UCHAR_MAX)
						throw runtime_error("InputReader::readFromStream: Maximal activity resource requirement is "+numberToStr(UCHAR_MAX)+"!");

					activitiesRequiredResources[actId][r] = (uint8_t) resourceReq;
				}
			}
			
			// Read resources capacity.
			for (uint8_t r = 0; r < totalNumberOfResources; ++r)	{
				uint32_t resourceCapacity = 0;
				if (!(IN>>resourceCapacity))
					throw runtime_error("InputReader::readFromStream: Invalid read of resource capacity!\nResource ID is "+numberToStr(r)+".");
				if (resourceCapacity > UCHAR_MAX)
					throw runtime_error("InputReader::readFromStream: Maximal capacity of resource is "+numberToStr(UCHAR_MAX)+"!");

				capacityOfResources[r] = (uint8_t) resourceCapacity;
			}
		} else {
			throw runtime_error("InputReader::readFromStream: Empty instance file??");
		}
	} else {
		throw runtime_error("InputReader::readFromStream: Parameter have to be file, not directory!"); 
	}

	// Check if resources are sufficient for activities.
	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)	{
		for (uint8_t resourceId = 0; resourceId < totalNumberOfResources; ++resourceId)	{
			if (activitiesRequiredResources[activityId][resourceId] > capacityOfResources[resourceId])
				throw runtime_error("InputReader::readFromStream: Suggested resources are insufficient for activities requirement!");
		}
	}
}

void InputReader::allocateBaseArrays()	{
	activitiesDuration = new uint8_t[numberOfActivities];
	activitiesSuccessors = new uint16_t*[numberOfActivities];
	activitiesNumberOfSuccessors = new uint16_t[numberOfActivities];
	capacityOfResources = new uint8_t[totalNumberOfResources];
	activitiesRequiredResources = new uint8_t*[numberOfActivities];
	for (uint16_t activityId = 0; activityId < numberOfActivities; ++activityId)
		activitiesRequiredResources[activityId] = new uint8_t[totalNumberOfResources];
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


uint32_t InputReader::strToNumber(const string& number)	const {
	istringstream istr(number,istringstream::in);
	uint32_t ret; istr>>ret;
	return ret;
}

string InputReader::numberToStr(const uint32_t& number) const {
	stringstream ss;
	ss<<number;
	return ss.str();
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


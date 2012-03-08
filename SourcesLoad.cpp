#include <algorithm>
#include <cstring>
#include "SourcesLoad.h"

using namespace std;

SourcesLoad::SourcesLoad(const uint8_t& numRes, const uint8_t * const& capRes) : numberOfResources(numRes), capacityOfResources(capRes)	{
	uint8_t maxCapacity = 0;
	resourcesLoad = new uint16_t*[numberOfResources];

	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		resourcesLoad[resourceId] = new uint16_t[capacityOfResources[resourceId]];
		memset(resourcesLoad[resourceId], 0, sizeof(uint16_t)*capacityOfResources[resourceId]);
		maxCapacity = max(capacityOfResources[resourceId],maxCapacity);
	}

	startValues = new uint16_t[maxCapacity];
	memset(startValues, 0, sizeof(uint16_t)*maxCapacity);
}

uint16_t SourcesLoad::getEarliestStartTime(const uint8_t * const& activityResourceRequirement)	const 	{
	uint16_t bestStart = 0;
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		uint8_t activityRequirement = activityResourceRequirement[resourceId];
		if (activityRequirement > 0)
			bestStart = max(resourcesLoad[resourceId][capacityOfResources[resourceId]-activityRequirement], bestStart);
	}
	return bestStart;
}

void SourcesLoad::addActivity(const uint16_t& activityStart, const uint16_t& activityStop, const uint8_t * const& activityRequirement)	{
	#if DEBUG_SOURCES == 1
	map<uint16_t,int16_t*>::iterator mit;
	if ((mit = peaks.find(activityStart)) == peaks.end())	{
		int16_t *peak = new int16_t[numberOfResources];
		for (uint8_t idx = 0; idx < numberOfResources; ++idx)
			peak[idx] = -((int16_t) activityRequirement[idx]);
		peaks[activityStart] = peak;
	} else {
		int16_t *peak = mit->second;
		for (uint8_t idx = 0; idx < numberOfResources; ++idx)
			peak[idx] -= activityRequirement[idx];
	}

	if ((mit = peaks.find(activityStop)) == peaks.end())	{
		int16_t *peak = new int16_t[numberOfResources];
		for (uint8_t idx = 0; idx < numberOfResources; ++idx)
			peak[idx] = ((int16_t) activityRequirement[idx]);
		peaks[activityStop] = peak;
	} else {
		int16_t *peak = mit->second;
		for (uint16_t idx = 0; idx < numberOfResources; ++idx)
			peak[idx] += activityRequirement[idx];
	}
	
	uint16_t **resourcesLoadCopy = new uint16_t*[numberOfResources];
	for (uint8_t i = 0; i < numberOfResources; ++i)	{
		resourcesLoadCopy[i] = new uint16_t[capacityOfResources[i]];
		for (uint8_t j = 0; j < capacityOfResources[i]; ++j)
			resourcesLoadCopy[i][j] = resourcesLoad[i][j];
	}	
	#endif

	int32_t requiredSquares, timeDiff;
	uint32_t k, c, capacityOfResource, resourceRequirement, baseResourceIdx, startTimePreviousUnit, newStartTime;
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		capacityOfResource = capacityOfResources[resourceId];
		resourceRequirement = activityRequirement[resourceId];
		requiredSquares = resourceRequirement*(activityStop-activityStart);
		if (requiredSquares > 0)	{
			baseResourceIdx = capacityOfResource-resourceRequirement;
			startTimePreviousUnit = ((resourceRequirement < capacityOfResource) ? resourcesLoad[resourceId][baseResourceIdx-1] : activityStop);
			newStartTime = min((uint32_t) activityStop, startTimePreviousUnit);
			if (activityStart < startTimePreviousUnit)	{
				for (k = baseResourceIdx; k < capacityOfResource; ++k)	{
					resourcesLoad[resourceId][k] = newStartTime; 
				}
				requiredSquares -= resourceRequirement*(newStartTime-activityStart); 
			}
			c = 0; k = 0;
			newStartTime = activityStop;
			while (requiredSquares > 0 && k < capacityOfResource)	{
				if (resourcesLoad[resourceId][k] < newStartTime)    {
					if (c >= resourceRequirement)
						newStartTime = startValues[c-resourceRequirement];
					timeDiff = newStartTime-max(resourcesLoad[resourceId][k],activityStart);
					if (requiredSquares-timeDiff > 0)	{
						requiredSquares -= timeDiff;
						startValues[c++] = resourcesLoad[resourceId][k];
						resourcesLoad[resourceId][k] = newStartTime; 
					} else {
						resourcesLoad[resourceId][k] = newStartTime-timeDiff+requiredSquares; 
						break;
					}
				}
				++k;
			}
		}
	}

	#if DEBUG_SOURCES == 1
	vector<uint16_t> tim;	
	vector<int16_t*> cum;	

	int16_t *currentLoad = new int16_t[numberOfResources];
	memset(currentLoad,0,sizeof(int16_t)*numberOfResources);

	for (map<uint16_t,int16_t*>::const_reverse_iterator it = peaks.rbegin(); it != peaks.rend(); ++it)	{
		int16_t *peak = it->second;	
		int16_t* cumVal = new int16_t[numberOfResources];
		for (uint8_t idx = 0; idx < numberOfResources; ++idx)	{
			currentLoad[idx] += peak[idx];
			cumVal[idx] = currentLoad[idx];
		}
		tim.push_back(it->first);
		cum.push_back(cumVal);
	}

	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		uint16_t pos = 0;
		uint8_t capacityOfResource = capacityOfResources[resourceId];
		uint16_t *startTimes = new uint16_t[capacityOfResource];
		memset(startTimes, 0, sizeof(uint16_t)*capacityOfResource);

		vector<uint16_t>::const_iterator it1 = tim.begin(), eit1 = tim.end();
		vector<int16_t*>::const_iterator it2 = cum.begin(), eit2 = cum.end();

		while (it1 != eit1 && it2 != eit2)	{
			uint16_t curTime = *it1;
			int16_t curVal = (*it2)[resourceId];
			if (curVal > ((int16_t) capacityOfResource))
				cerr<<"Overload resource "<<resourceId+1<<endl;
			while (curVal > ((int16_t) pos) && ((int16_t) pos) < capacityOfResource)	{
				startTimes[pos++] = curTime;
			}

			++it1; ++it2;
		}

		bool correct = true;
		for (uint8_t j = 0; j < capacityOfResource; ++j)	{
			if (startTimes[j] != resourcesLoad[resourceId][j])	{
				cerr<<"SourcesLoad::addActivity: Please fix computation of vector."<<endl;
				correct = false; break;
			}
		}

		if (!correct)	{
			cerr<<"Resource id: "<<resourceId<<endl;
			cerr<<"activity times: "<<activityStart<<" "<<activityStop<<endl;
			cerr<<"activity requirement: "<<activityRequirement[resourceId]<<endl;
			cerr<<"Original start times vector: "<<endl;
			for (uint8_t i = 0; i < capacityOfResource; ++i)	{
				cerr<<" "<<resourcesLoadCopy[resourceId][i];
			}
			cerr<<endl;
			cerr<<"Probably correct result: "<<endl;
			for (uint8_t i = 0; i < capacityOfResource; ++i)	{
				cerr<<" "<<startTimes[i];
			}
			cerr<<endl;
			cerr<<"Probably incorrect result: "<<endl;
			for (uint8_t i = 0; i < capacityOfResource; ++i)	{
				cerr<<" "<<resourcesLoad[resourceId][i];
			}
			cerr<<endl;
		}
		
		delete[] startTimes;
	}

	for (vector<int16_t*>::const_iterator it = cum.begin(); it != cum.end(); ++it)
		delete[] *it;
	delete[] currentLoad;

	for (uint8_t i = 0; i < numberOfResources; ++i)
		delete[] resourcesLoadCopy[i];
	delete[] resourcesLoadCopy;
	#endif
}

void SourcesLoad::printCurrentState(ostream& OUT)	const	{
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		OUT<<"Resource "<<(uint16_t) resourceId+1<<":";
		for (uint8_t capIdx = 0; capIdx < capacityOfResources[resourceId]; ++capIdx)	
			OUT<<" "<<resourcesLoad[resourceId][capIdx];
		OUT<<endl;
	}
}

SourcesLoad::~SourcesLoad()	{
	for (uint16_t** ptr = resourcesLoad; ptr < resourcesLoad+numberOfResources; ++ptr)
		delete[] *ptr;
	delete[] resourcesLoad;
	delete[] startValues;
	#if DEBUG_SOURCES == 1
	for (map<uint16_t,int16_t*>::const_iterator mit = peaks.begin(); mit != peaks.end(); ++mit)
		delete[] mit->second;
	#endif
}


#include <algorithm>
#include <cstring>
#include "SourcesLoad.h"

using namespace std;

SourcesLoad::SourcesLoad(const uint8_t numRes, uint8_t *capRes) : numberOfResources(numRes), capacityOfResources(capRes)	{
	uint8_t maxCapacity = 0;
	resourcesLoad = new uint16_t*[numberOfResources];

	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		resourcesLoad[resourceId] = new uint16_t[capacityOfResources[resourceId]];
		memset(resourcesLoad[resourceId], 0, sizeof(uint16_t)*capacityOfResources[resourceId]);
		maxCapacity = max(capacityOfResources[resourceId],maxCapacity);
	}

	startValues = new uint16_t[maxCapacity];
	memset(startValues, 0, sizeof(uint16_t)*maxCapacity);
	reqItems = new uint16_t[maxCapacity];
	memset(reqItems, 0, sizeof(uint16_t)*maxCapacity);
}

uint16_t SourcesLoad::getEarliestStartTime(uint8_t *activityResourceRequirement)	const 	{
	uint16_t bestStart = 0;
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		uint8_t activityRequirement = activityResourceRequirement[resourceId];
		if (activityRequirement > 0)
			bestStart = max(resourcesLoad[resourceId][activityRequirement-1], bestStart);
	}
	return bestStart;
}

void SourcesLoad::addActivity(uint16_t activityStart, uint16_t activityStop, uint8_t *activityRequirement)	{
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
	#endif

	bool writeValue, workDone;
	uint16_t sourceReq, curLoad, idx;
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		sourceReq = activityRequirement[resourceId];
		for (uint8_t capIdx = capacityOfResources[resourceId]; capIdx > 0; --capIdx)	{
			curLoad = resourcesLoad[resourceId][capIdx-1];
			if (sourceReq > 0)	{
				if (curLoad <= activityStart)	{
					resourcesLoad[resourceId][capIdx-1] = activityStop;
					--sourceReq;
					idx = 0;
					while (startValues[idx] != 0)	{
						if (reqItems[idx] > 0)
							--reqItems[idx];
						++idx;
					}
				} else if (curLoad < activityStop)	{
					resourcesLoad[resourceId][capIdx-1] = activityStop;
					--sourceReq;
					idx = 0;
					writeValue = true;
					while (startValues[idx] != 0)	{
						if (startValues[idx] > curLoad && reqItems[idx] > 0)
							--reqItems[idx];
						if (startValues[idx] == curLoad)
							writeValue = false;
						++idx;
					}
					if (writeValue == true && curLoad > 0)	{
						startValues[idx] = curLoad;
						reqItems[idx] = activityRequirement[resourceId];
					}
				}
			} else {
				idx = 0;
				workDone = true;
				while (startValues[idx] != 0)	{
					if (reqItems[idx] != 0)	{
						workDone = false;
						break;
					}
					++idx;
				}
				if (workDone == true)	{
					break;
				} else {
					writeValue = true;
					resourcesLoad[resourceId][capIdx-1] = startValues[idx];
					idx = 0;
					while (startValues[idx] != 0)	{
						if (curLoad < startValues[idx] && reqItems[idx] > 0)
							--reqItems[idx];
						if (curLoad == startValues[idx])
							writeValue = false;
						++idx;
					}
					if (writeValue == true && curLoad > activityStart)	{
						startValues[idx] = curLoad;
						reqItems[idx] = activityRequirement[resourceId];
					}
				}
			}
		}
		idx = 0;
		while (startValues[idx] != 0)
			startValues[idx++] = 0;
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
		reverse(startTimes, startTimes+capacityOfResource);

		bool correct = true;
		for (uint8_t j = 0; j < capacityOfResource; ++j)	{
			if (startTimes[j] != resourcesLoad[resourceId][j])	{
				cerr<<"SourcesLoad::addActivity: Please fix computation of vector."<<endl;
				correct = false; break;
			}
		}

		if (!correct)	{
			cerr<<"Resource id: "<<resourceId<<endl;
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
	#endif
}

void SourcesLoad::printCurrentState(ostream& OUT)	const	{
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		OUT<<"Resource "<<resourceId+1<<":";
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
	delete[] reqItems;
	#if DEBUG_SOURCES == 1
	for (map<uint16_t,int16_t*>::const_iterator mit = peaks.begin(); mit != peaks.end(); ++mit)
		delete[] mit->second;
	#endif
}


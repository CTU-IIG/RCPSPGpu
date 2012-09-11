#include "SourcesLoad.h"

using namespace std;

SourcesLoad::SourcesLoad(const uint8_t& numberOfResources, const uint8_t * const& capacitiesOfResources, const uint16_t& makespanUpperBound)
       	: numberOfResources(numberOfResources), capacitiesOfResources(capacitiesOfResources), makespanUpperBound(makespanUpperBound)	{
	remainingResourcesCapacity = new uint8_t*[numberOfResources];
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		remainingResourcesCapacity[resourceId] = new uint8_t[makespanUpperBound];
		for (uint16_t t = 0; t < makespanUpperBound; ++t)
			remainingResourcesCapacity[resourceId][t] = capacitiesOfResources[resourceId];
	}
}

uint16_t SourcesLoad::getEarliestStartTime(const uint8_t * const& activityResourceRequirements,
		const uint16_t& earliestPrecedenceStartTime, const uint8_t& activityDuration)	const 	{
	uint16_t loadTime = 0, t = makespanUpperBound;
	for (t = earliestPrecedenceStartTime; t < makespanUpperBound && loadTime < activityDuration; ++t)	{
		bool capacityAvailable = true;
		for (uint8_t resourceId = 0; resourceId < numberOfResources && capacityAvailable; ++resourceId)	{
			if (remainingResourcesCapacity[resourceId][t] < activityResourceRequirements[resourceId])	{
				loadTime = 0;
				capacityAvailable = false;
			}
		}
		if (capacityAvailable)
			++loadTime;
	}
	return t-loadTime;
}

void SourcesLoad::addActivity(const uint16_t& activityStart, const uint16_t& activityStop, const uint8_t * const& activityRequirements)	{
	for (uint8_t resourceId = 0; resourceId < numberOfResources; ++resourceId)	{
		for (uint16_t t = activityStart; t < activityStop; ++t)	{
			remainingResourcesCapacity[resourceId][t] -= activityRequirements[resourceId];
		}
	}
}

SourcesLoad::~SourcesLoad()	{
	for (uint8_t** ptr = remainingResourcesCapacity; ptr < remainingResourcesCapacity+numberOfResources; ++ptr)
		delete[] *ptr;
	delete[] remainingResourcesCapacity;
}


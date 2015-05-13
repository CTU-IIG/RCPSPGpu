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


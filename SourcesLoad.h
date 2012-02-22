#ifndef HLIDAC_PES_SOURCES_LOAD_H
#define HLIDAC_PES_SOURCES_LOAD_H

#define DEBUG_SOURCES 0

#include <iostream>
#include <stdint.h>

#if DEBUG_SOURCES == 1
#include <map>
#include <vector>
#endif

class SourcesLoad {
	public:
		SourcesLoad(const uint8_t numRes, uint8_t *capRes);

		uint16_t getEarliestStartTime(uint8_t *activityResourceRequirement) const;
		void addActivity(uint16_t activityStart, uint16_t activityStop, uint8_t *activityRequirement);
		void printCurrentState(std::ostream& OUT = std::cout) const;

		~SourcesLoad();

	private:

		/* COPY OF OBJECT IS FORBIDDEN */

		SourcesLoad(const SourcesLoad&);
		SourcesLoad& operator=(const SourcesLoad&);


		// Total number of resources.
		uint8_t numberOfResources;
		// Capacity of resources.
		uint8_t *capacityOfResources;
		// Current state of resources.
		uint16_t **resourcesLoad;
		// Helper arrays.
		uint16_t *startValues, *reqItems;

		#if DEBUG_SOURCES == 1
		std::map<uint16_t,int16_t*> peaks;
		#endif
};

#endif


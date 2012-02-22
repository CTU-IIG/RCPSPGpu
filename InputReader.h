#ifndef HLIDAC_PES_INPUT_READER_H
#define HLIDAC_PES_INPUT_READER_H

#include <iostream>
#include <string>
#include <vector>
#include <stdint.h>

class InputReader {
	public:
		InputReader() : numberOfActivities(0), totalNumberOfResources(0), activitiesDuration(NULL), activitiesRequiredResources(NULL),
						activitiesSuccessors(NULL), activitiesNumberOfSuccessors(NULL), capacityOfResources(NULL) { }

		void readFromFile(const std::string& filename);
		void readFromStream(std::istream& IN);

		uint16_t getNumberOfActivities() const { return numberOfActivities; }
		uint8_t getNumberOfResources() const { return totalNumberOfResources; }
		uint8_t* getActivitiesDuration() const { return activitiesDuration; }
		uint8_t** getActivitiesResources() const { return activitiesRequiredResources; }
		uint16_t** getActivitiesSuccessors() const { return activitiesSuccessors; }
		uint16_t* getActivitiesNumberOfSuccessors() const { return activitiesNumberOfSuccessors; }
		uint8_t* getCapacityOfResources() const { return capacityOfResources; }

		void printInstance(std::ostream& OUT = std::cout)	const;

		~InputReader() { freeInstanceData(); }

	protected:
		uint16_t strToNumber(const std::string& number)	const;

	private:

		void freeInstanceData();

		InputReader(const InputReader&);
		InputReader& operator=(const InputReader&);

		/* INSTANCE DATA */
		uint16_t numberOfActivities;
		uint8_t totalNumberOfResources;
		uint8_t *activitiesDuration;
		uint8_t **activitiesRequiredResources;
		uint16_t **activitiesSuccessors;
		uint16_t *activitiesNumberOfSuccessors;
		uint8_t *capacityOfResources;

		friend void writeHeaderFile(const std::vector<std::string>& inputFiles, const std::string& headerFile);
};

void writeHeaderFile(const std::vector<std::string>& inputFiles, const std::string& headerFile);

#endif


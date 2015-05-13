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
#ifndef HLIDAC_PES_SOURCES_LOAD_H
#define HLIDAC_PES_SOURCES_LOAD_H

/*!
 * \file SourcesLoad.h
 * \author Libor Bukata
 * \brief Implementation of SourcesLoad class.
 */

#include <iostream>
#include <stdint.h>

/*!
 * \class SourcesLoad
 * \brief Implementation of resources evaluation. For each time unit available free capacity is remembered.
 */
class SourcesLoad {
	public:
		/*!
		 * \param numberOfResources Number of renewable resources with constant capacity.
		 * \param capacitiesOfResources Maximal capacity of each resource.
		 * \param makespanUpperBound Estimate of maximal project duration.
		 * \brief It allocates required data-structures and fill them with initial values.
		 */
		SourcesLoad(const uint8_t& numberOfResources, const uint8_t * const& capacitiesOfResources, const uint16_t& makespanUpperBound);

		/*!
		 * \param activityResourceRequirements Activity requirement for each resource.
		 * \param earliestPrecedenceStartTime The earliest activity start time without precedence violation.
		 * \param activityDuration Duration of the activity.
		 * \return The earliest activity start time without precedence and resources violation.
		 */
		uint16_t getEarliestStartTime(const uint8_t * const& activityResourceRequirements,
			       	const uint16_t& earliestPrecedenceStartTime, const uint8_t& activityDuration) const;
		/*!
		 * \param activityStart Start time of the scheduled activity.
		 * \param activityStop Finish time of the scheduled activity.
		 * \param activityRequirements Activity requirement for each resource.
		 * \brief Update state of resources with respect to the added activity.
		 */
		void addActivity(const uint16_t& activityStart, const uint16_t& activityStop, const uint8_t * const& activityRequirements);

		//! Free allocated memory.
		~SourcesLoad();

	private:

		//! Copy constructor is forbidden.
		SourcesLoad(const SourcesLoad&);
		//! Assignment operator is forbidden.
		SourcesLoad& operator=(const SourcesLoad&);

		//! Number of renewable resources with constant capacity.
		const uint8_t numberOfResources;
		//! Capacities of the resources.
		const uint8_t * const capacitiesOfResources;
		//! Upper bound of the project duration.
		const uint16_t makespanUpperBound;
		//! Available capacity for each resource (independent variable is time).
		uint8_t **remainingResourcesCapacity;
};

#endif


#include "stdafx.h"
#include "utils.h"

// page 49
float GetNewOrientation(const float &currOrientation, const VECTOR2 &velocity)
{
	// Make sure we have a velocity
	if (velocity.Norm() > 0)
	{
		// Calculate orientation using an arc tangent of
		// the velocity components.
		return atan2(velocity.y, velocity.x);
	}
	else
	{
		// Otherwise use the current orientation
		return currOrientation;
	}
}


/**
 * page 54
 * @brief returns a random value in range [-1,1] s.t.
 * numbers near 0 are more likely * 
 * @return float 
 */
float RandomBinomial()
{
	float randomVal1 = static_cast<float>(rand()) / RAND_MAX;
	float randomVal2 = static_cast<float>(rand()) / RAND_MAX;
	return randomVal1 - randomVal2;
}
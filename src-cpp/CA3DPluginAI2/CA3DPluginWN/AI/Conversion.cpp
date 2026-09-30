#include "stdafx.h"
#include "Conversion.h"

/**
 * @brief computes the orientation vector for a given orientation angle
 * 	      assuming a right-handed coordinate system
 * 
 * @param angle (rad) the input angle
 * @return VECTOR2 the vector corresponding to the given angle
 */
VECTOR2 OrientationAngle2Vector(float angle)
{
	VECTOR2 orientVec;
	orientVec.x = sin(angle);
	orientVec.y = cos(angle);
	return orientVec;
}
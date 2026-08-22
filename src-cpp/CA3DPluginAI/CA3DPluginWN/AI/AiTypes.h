#pragma once
#include "../CA3D_math/Types.h"

/*
The name 'Static' stems from the fact that this data
structure doesn't contain any info about the character's motion
*/
struct Static
{
	VECTOR2 position; // 2D vector
	float orientation; // the direction in which a character is facing (rad)
};

/* 
The set of accelerations returned by a steering behavior in
order to control velocities of a character
*/
struct SteeringOutput2D
{
	VECTOR2 linearAcc;
	float angularAcc;
};

/*
The 3D version of SteeringOutput2D
*/
struct SteeringOutput3D
{
	VECTOR3 linearAcc;
	float angularAcc;
};

// 2D kinematic of a character
struct Kinematic2D
{
	VECTOR2 position;  // 2D position
	float orientation; // orienation angle (rad)
	VECTOR2 velocity;  // linear velocity
	float rotation;	   /* angular velocity (rad/sec): represents how fast the
						  character's orientation is changing */

	void update(SteeringOutput2D steering, float time)
	{page 47...
		// Update the position and orientation
		position += velocity * time +
				0.5 * steering.linearAcc * time;
		orientation += rotation * time +
				0.5 * steering.angularAcc * time * time;
				
		// and the velocity and rotation
		velocity += steering.linearAcc * time;
		orientation += steering.angularAcc * time;
	}
};

// 3D kinematic of a character
struct Kinematic3D
{
	VECTOR3 position;  // 3D position
	float orientation; // orienation angle (rad)
	VECTOR3 velocity;  // linear velocity
	float rotation;	   /* angular velocity (rad/sec): represents how fast the
						  character's orientation is changing */
};

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
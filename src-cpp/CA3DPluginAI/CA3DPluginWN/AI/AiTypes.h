#include "../CA3D_math/Types.h"

/*
The name 'Static' stems from the fact that this data
structure doesn't contain any info about the character's motion
*/
struct Static
{
	VECTOR2 position; // 2D vector
	float orientation; // the direction in which a character is facing (rad)

	Static()
	{
		position.x = 0;
		position.y = 0;
		orientation = 0;
	}

	Static(VECTOR2 position_, float orientation_)
	{
		position = position_;
		orientation = orientation_;
	}

	// page 44: conversion of orientation angle to vector
	VECTOR2 OrientationAsVector()
	{
		// a right-handed coordinate system is assumed
		VECTOR2 orientVec;
		orientVec.x = sin(orientation);
		orientVec.y = cos(orientation);
		return orientVec;
	}	
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




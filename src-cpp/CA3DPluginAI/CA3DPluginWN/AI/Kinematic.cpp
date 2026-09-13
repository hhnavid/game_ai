#include "stdafx.h"
#include "Kinematic.h"

KinematicBehavior::KinematicBehavior(Static character_, Static target_, float maxSpeed_)
{
	character = character_;
	target = target_;
	maxSpeed = maxSpeed_;
}

KinematicSeek::KinematicSeek(Static character_, Static target_, float maxSpeed_):
	KinematicBehavior(character_, target_, maxSpeed_)
{}

KinematicSteerOut2D KinematicSeek::GetSteering()
{
	// Create the structure for output
	KinematicSteerOut2D steering;		 
		
	// Get the direction to the target
	steering.velocity = target.position - character.position;
		
	// The velocity is along this direction, at full speed
	steering.velocity.Normalize();
	steering.velocity *= maxSpeed;
		
	// Face in the direction we want to move
	character.orientation = getNewOrientation(character.orientation,
										      steering.velocity);
	// Output the steering	
	steering.rotation = 0;
	return steering;
}

// page 49
float getNewOrientation(const float &currOrientation, const VECTOR2 &velocity)
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
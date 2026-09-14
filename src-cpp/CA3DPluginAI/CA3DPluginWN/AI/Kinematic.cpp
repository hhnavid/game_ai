#include "stdafx.h"
#include "Kinematic.h"

KinematicBehavior::KinematicBehavior(Static character_, Static target_, float maxSpeed_)
{
	character = character_;
	target = target_;
	maxSpeed = maxSpeed_;
}

SeekBehavior::SeekBehavior(Static character_, Static target_, float maxSpeed_) : KinematicBehavior(character_, target_, maxSpeed_)
{
}

KinematicSteerOut2D SeekBehavior::GetSteering()
{
	// Create the structure for output
	KinematicSteerOut2D steering;

	// Get the direction to the target
	steering.velocity = target.position - character.position;

	// The velocity is along this direction, at full speed
	steering.velocity.Normalize();
	steering.velocity *= maxSpeed;

	// Face in the direction we want to move
	character.orientation = GetNewOrientation(character.orientation,
											  steering.velocity);
	// Output the steering
	steering.rotation = 0;
	return steering;
}

FleeBehavior::FleeBehavior(Static character_, Static target_, float maxSpeed_) : KinematicBehavior(character_, target_, maxSpeed_)
{
}

KinematicSteerOut2D FleeBehavior::GetSteering()
{
	// Create the structure for output
	KinematicSteerOut2D steering;

	// Get the direction to the target
	steering.velocity = character.position - target.position;

	// The velocity is along this direction, at full speed
	steering.velocity.Normalize();
	steering.velocity *= maxSpeed;

	// Face in the direction we want to move
	character.orientation = GetNewOrientation(character.orientation,
											  steering.velocity);
	// Output the steering
	steering.rotation = 0;
	return steering;
}

ArriveBehavior::ArriveBehavior(Static character_, Static target_, float maxSpeed_, float radius_, float timeToTarget_) : KinematicBehavior(character_, target_, maxSpeed_)
{
	radius = radius_;
	timeToTarget = timeToTarget_;
}

KinematicSteerOut2D ArriveBehavior::GetSteering()
{
	// Create the structure for output
	KinematicSteerOut2D steering;

	// Get the direction to the target
	steering.velocity = target.position - character.position;

	// Check if we're close enough to the target
	if (steering.velocity.Norm() < radius)
	{
		// Yap! we're close enough stop the seeking
		steering.rotation = 0;
		steering.velocity = VECTOR2(0, 0);
		return steering;
	}

	// keep moving toward the target but keep the
	// speed proportional to the distance to target
	steering.velocity /= timeToTarget;

	// clip the velocity to its max value if needed
	if (steering.velocity.Norm() > maxSpeed)
	{
		steering.velocity.Normalize();
		steering.velocity *= maxSpeed;
	}
	// Face in the direction we want to move
	character.orientation = GetNewOrientation(character.orientation,
											  steering.velocity);
	// Output the steering
	steering.rotation = 0;
	return steering;
}

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
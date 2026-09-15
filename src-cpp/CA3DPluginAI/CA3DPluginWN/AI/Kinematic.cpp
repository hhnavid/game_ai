#include "stdafx.h"
#include "Kinematic.h"

KinematicBehavior::KinematicBehavior(Static character_, float maxSpeed_)
{
	character = character_;	
	maxSpeed = maxSpeed_;
}

KinematicBehavior::~KinematicBehavior()
{}

KinematicSeek::KinematicSeek(Static character_, Static target_, float maxSpeed_) : KinematicBehavior(character_, maxSpeed_)
{
	target = target_;
}

KinematicSeek::~KinematicSeek()
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
	character.orientation = GetNewOrientation(character.orientation,
											  steering.velocity);
	// Output the steering
	steering.rotation = 0;
	return steering;
}

KinematicFlee::KinematicFlee(Static character_, Static target_, float maxSpeed_) : KinematicBehavior(character_, maxSpeed_)
{
	target = target_;
}

KinematicFlee::~KinematicFlee()
{}

KinematicSteerOut2D KinematicFlee::GetSteering()
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

KinematicArrive::KinematicArrive(Static character_, Static target_, float maxSpeed_, float radius_, float timeToTarget_) :
	KinematicBehavior(character_, maxSpeed_)
{
	target = target_;
	radius = radius_;
	timeToTarget = timeToTarget_;
}

KinematicArrive::~KinematicArrive()
{}

KinematicSteerOut2D KinematicArrive::GetSteering()
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

KinematicWander::KinematicWander(Static character_, float maxSpeed_, float maxRotation_) :
	KinematicBehavior(character_, maxSpeed_)
{
	maxRotation = maxRotation_;
}

KinematicWander::~KinematicWander()
{}

KinematicSteerOut2D KinematicWander::GetSteering()
{
	// Create the structure for output
	KinematicSteerOut2D steering;

	// Get velocity from the vector form of the orientation
	steering.velocity = maxSpeed * character.OrientationAsVector();

	// Change the steering orientation randomly to create a wandering motion
	steering.rotation = RandomBinomial() * maxRotation;

	// Output the steering
	return steering;
}
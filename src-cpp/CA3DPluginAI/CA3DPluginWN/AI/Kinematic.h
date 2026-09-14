#include "AiTypes.h"

/*
Updating a character's position and orientation is usually done
by the physics simulation layer of the game engine. However, if
you need to update them manually, use Kinematic2/3D structs below.
*/

// 2D kinematic of a character (page 47)
struct Kinematic2D
{
	VECTOR2 position;  // 2D position
	float orientation; // orienation angle (rad)
	VECTOR2 velocity;  // linear velocity
	float rotation;	   /* angular velocity (rad/sec): represents how fast the
						  character's orientation is changing */

	void Update(SteeringOutput2D &steering, float time)
	{
		// Update the position and orientation
		position += velocity * time;	// at^2 term is negligible due to time steps being small
		orientation += rotation * time; // wt^2 term is negligible due to time steps being small

		// and the velocity and rotation
		velocity += steering.linearAcc * time;
		orientation += steering.angularAcc * time;
	}
};

// 3D kinematic of a character (page 47)
struct Kinematic3D
{
	VECTOR3 position;  // 3D position
	float orientation; // orienation angle (rad)
	VECTOR3 velocity;  // linear velocity
	float rotation;	   /* angular velocity (rad/sec): represents how fast the
						  character's orientation is changing */

	void Update(SteeringOutput3D &steering, float time)
	{
		// Update the position and orientation
		position += 0.5 * steering.linearAcc * time * time + velocity * time;
		orientation += 0.5 * steering.angularAcc * time * time + rotation * time;

		// and the velocity and rotation
		velocity += steering.linearAcc * time;
		orientation += steering.angularAcc * time;
	}
};

/*page50*/
struct KinematicSteerOut2D
{
	VECTOR2 velocity; // the steering direction & speed
	float rotation;	  // amount of rotation (rad) due to steering
};

class KinematicBehavior
{
public:
	KinematicBehavior(Static character_, Static target_, float maxSpeed_);
	virtual KinematicSteerOut2D GetSteering() = 0;
	inline void setTarget(Static target_) { target = target_; }
	inline Static getTarget() { return target; }
	inline void setCharacter(Static character_) { character = character_; }
	inline Static getCharacter() { return character; }

protected:
	float maxSpeed;
	Static character; // Pose of the character
	/* Pose of the target. Depending on the child class implementation,
	   the character may for example seek the target or run from it.
	*/
	Static target;
};

/*page 50*/
class SeekBehavior : public KinematicBehavior
{
public:
	SeekBehavior(Static character_, Static target_, float maxSpeed_);
	virtual KinematicSteerOut2D GetSteering();
};

/* page 51 */
class FleeBehavior : public KinematicBehavior
{
public:
	FleeBehavior(Static character_, Static target_, float maxSpeed_);
	virtual KinematicSteerOut2D GetSteering();
};

/**
 * page52: Arrive behavior is similar to the Seek behavior except that it uses timeToTarget
 * & radius attributes to avoid wiggling motion when the character gets too close to the target.
 */
class ArriveBehavior : public KinematicBehavior
{
public:
	ArriveBehavior(Static character_, Static target_, float maxSpeed_, float radius_, float timeToTarget_);
	virtual KinematicSteerOut2D GetSteering();
private:
	/* when the character distance to the target < radius,
	 the target is reached & seeking is over.*/
	float radius;

	/* time to target constant (in seconds), used to slow down the character's movement
	as it gets closer to the target.*/
	float timeToTarget = 0.25;
};

float GetNewOrientation(const float &currOrientation, const VECTOR2 &velocity);
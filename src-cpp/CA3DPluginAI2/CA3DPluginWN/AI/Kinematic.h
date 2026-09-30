#include "utils.h"

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
	KinematicBehavior(Static character_, float maxSpeed_);
	~KinematicBehavior();
	virtual KinematicSteerOut2D GetSteering() = 0;	
	inline void setCharacter(Static character_) { character = character_; }
	inline Static getCharacter() { return character; }
	// TODO: add set/get methods for all of the class attributes
protected:
	float maxSpeed;
	Static character; // Pose of the character	
};

/*page 50*/
class KinematicSeek : public KinematicBehavior
{
public:
	KinematicSeek(Static character_, Static target_, float maxSpeed_);
	~KinematicSeek();
	virtual KinematicSteerOut2D GetSteering();
	inline void setTarget(Static target_) { target = target_; }
	inline Static getTarget() { return target; }

private:
	// Pose of the target sought by the character.	
	Static target;
};

/* page 51 */
class KinematicFlee : public KinematicBehavior
{
public:
	KinematicFlee(Static character_, Static target_, float maxSpeed_);
	~KinematicFlee();
	virtual KinematicSteerOut2D GetSteering();
	inline void setTarget(Static target_) { target = target_; }
	inline Static getTarget() { return target; }
private:
	// Pose of the target from which the character tries to flee.
	Static target;
};

/**
 * page52: Arrive behavior is similar to the Seek behavior except that it uses timeToTarget
 * & radius attributes to avoid wiggling motion when the character gets too close to the target.
 */
class KinematicArrive : public KinematicBehavior
{
public:
	KinematicArrive(Static character_, Static target_, float maxSpeed_, float radius_, float timeToTarget_);
	~KinematicArrive();
	virtual KinematicSteerOut2D GetSteering();
	inline void setTarget(Static target_) { target = target_; }
	inline Static getTarget() { return target; }
	// TODO: add set/get methods for all of the class attributes
private:
	// Pose of the target sought by the character.	
	Static target;

	/* when the character distance to the target < radius,
	 the target is reached & seeking is over.*/
	float radius;

	/* time to target constant (in seconds), used to slow down the character's movement
	as it gets closer to the target.*/
	float timeToTarget = 0.25;
};

/**
 * page 53: Wandering behavior in which the character always move in the direction it's facing.
 * The direction of the character is changed randomly to create a wandering effect. 
 */
class KinematicWander : public KinematicBehavior
{
public:
	KinematicWander(Static character_, float maxSpeed_, float maxRotation_);
	~KinematicWander();
	virtual KinematicSteerOut2D GetSteering();

private:
	/* The max rotation speed which should be
	set smaller than the maximum possible value so 
	that a liesurely change in the direction will be possible*/
	float maxRotation;
};


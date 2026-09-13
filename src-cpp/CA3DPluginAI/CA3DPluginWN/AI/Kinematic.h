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
		position += velocity * time; // at^2 term is negligible due to time steps being small
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
	float rotation; // amount of rotation (rad) due to steering
};

class KinematicBehavior
{
public:
	KinematicBehavior(Static character_, Static target_, float maxSpeed_);
	virtual KinematicSteerOut2D GetSteering() = 0;	
protected:
	float maxSpeed;
	Static character; // Pose of the character
	/* Pose of the target. Depending on the child class implementation,
	   the character may for example seek the target or run from it.
	*/
	Static target;    
};

/*page 50*/
class KinematicSeek : public KinematicBehavior
{
public:
	KinematicSeek(Static character_, Static target_, float maxSpeed_);
	virtual KinematicSteerOut2D GetSteering();
	inline void setTarget(Static target_) { target = target_; }
	inline Static getTarget() { return target; }	
	inline void setCharacter(Static character_) { character = character_; }
	inline Static getCharacter() { return character; }
};

float getNewOrientation(const float &currOrientation, const VECTOR2 &velocity);
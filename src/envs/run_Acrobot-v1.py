# this script runs AirRaid-ram-v0 env with random actions just for
# illustation purposes
import gymnasium as gym
import time

   
# Create the environment
env = gym.make("Acrobot-v1", render_mode="human")

# Seed for reproducibility (optional)
env.action_space.seed(42)

# Reset environment to start state
observation, info = env.reset(seed=42)

# Run the environment for a fixed number of steps
for _ in range(1000):
    # Sample a random action from the action space
    action = env.action_space.sample()
    # Take the step with the random action
    observation, reward, terminated, truncated, info = env.step(action)

    # Render the environment
    env.render()

    # Reset the environment when done
    if terminated or truncated:
        observation, info = env.reset()

    # Small delay to slow down rendering for visualization purpose
    time.sleep(0.01)

# Close the environment when done
env.close()

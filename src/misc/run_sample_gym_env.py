
import gym

# Create the environment (e.g., CartPole-v1)
env = gym.make("CartPole-v1", render_mode="human")
state = env.reset(seed=42)  # Reset environment; returns initial observation

num_steps = 20  # Number of steps to run

for step in range(num_steps):
    action = env.action_space.sample()  # Sample a random action
    next_state, reward, done, info, _ = env.step(action)  # Take action
    env.render()

    print(f"Step {step}: Reward = {reward}, Next state = {next_state}")

    if done:
        print("Episode finished after {} steps.".format(step + 1))
        state = env.reset()
        break  # Optional: stop after first episode ends

env.close()

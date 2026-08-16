# this script runs AirRaid-ram-v0 env with random actions just for
# illustation purposes
import sys
sys.path.append('I:/projs/game-ai/src')

import time
import gymnasium as gym
# from ..common.replay_buffer import ReplayBuffer
# from common.stbl3_buffers import NStepReplayBuffer
# from common.stbl3_replay_buffers import NS
   
# Create the environment
env = gym.make("Acrobot-v1", render_mode="human")

# Seed for reproducibility (optional)
env.action_space.seed(42)

# Reset environment to start state
s0, info = env.reset()

# memory = NStepReplayBuffer(n_steps=5, gamma=0.95)

# Run the environment for a fixed number of steps
for _ in range(72):
    # Sample a random action from the action space
    a = env.action_space.sample()
    # Take the step with the random action
    s1, r, terminated, truncated, info = env.step(a)

    # Render the environment
    # env.render()

    done = terminated or truncated
    # memory.append(prev_state=s0,
    #               next_state=s1,
    #               action=a,     
    #               reward=r, 
    #               done=done)    ???
    # memory.add(s0, s1, a, r, )
    
    # Reset the environment when done
    if terminated or truncated:
        s0, info = env.reset()
        done = False
    else:
        s0 = s1

    # Small delay to slow down rendering for visualization purpose
    # time.sleep(0.01)        

# Close the environment when done
env.close()

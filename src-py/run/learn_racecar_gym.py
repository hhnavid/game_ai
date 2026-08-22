from time import sleep
import gymnasium
from racecar_gym.envs import gym_api

env = gymnasium.make(
    id='MultiAgentRaceEnv-v0',
    scenario='/media/navid/game/projs/ai-agents/racecar_gym/examples/scenarios/custom.yml',
    render_mode='human'
)

print(env.observation_space)
print(env.action_space)

done = False
obs = env.reset(options=dict(mode='grid'))

while not done:
    action = env.action_space.sample()
    obs, rewards, dones, truncated, states = env.step(action)
    done = any(dones.values())
    env.render()
    sleep(0.01)

env.close()


# import gymnasium
# import racecar_gym.envs.gym_api

# # For predefined environments:
# env = gymnasium.make(
#     id='SingleAgentAustria-v0',
#     render_mode='human'
# )

# # For custom scenarios:
# env = gymnasium.make(
#     id='SingleAgentRaceEnv-v0', 
#     scenario='/media/navid/game/projs/ai-agents/racecar_gym/scenarios/austria.yml',
#     render_mode='rgb_array_follow', # optional
#     render_options=dict(width=320, height=240, agent='A') # optional
# )

# done = False
# reset_options = dict(mode='grid')
# obs, info = env.reset(options=reset_options)

# while not done:
#     action = env.action_space.sample()
#     obs, rewards, terminated, truncated, states = env.step(action)
#     done = terminated or truncated
#     env.render(mode='human')

# env.close()

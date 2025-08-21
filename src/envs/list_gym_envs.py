import gymnasium as gym

# list all of the available envs from gymnasium
for env_id in gym.envs.registry.keys():
    print(env_id)
    
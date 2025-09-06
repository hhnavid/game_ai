import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

# Create the environment
env = gym.make("Acrobot-v1")

# Instantiate the DQN model with MLP policy
model = DQN("MlpPolicy", env, verbose=1, batch_size=128, buffer_size=50000,
            learning_rate=0.00063, gamma=0.99, train_freq=4, target_update_interval=250,
            exploration_fraction=0.12, exploration_final_eps=0.1,
            policy_kwargs=dict(net_arch=[256, 256]))

# Train the model
model.learn(total_timesteps=300000,#100000,
            progress_bar=True)

# Save the trained model
model.save("dqn_acrobot")

# Evaluate the trained agent
print("evaluating learned policy...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Enjoy the trained agent (optional)
eval_env = gym.make("Acrobot-v1", render_mode="human")
obs, info = eval_env.reset()
for _ in  tqdm(range(1000)):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    if terminated or truncated:
        obs, info = eval_env.reset()

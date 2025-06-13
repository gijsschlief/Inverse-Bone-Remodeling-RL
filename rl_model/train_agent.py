import gym
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from rl_model.reward_calculation import calculate_similarity

# Define your custom reward calculation function
def reward_calculate(observation, action, next_observation):
    # Replace this with your custom reward logic
    reward = -abs(next_observation[0])  # Example: penalize distance from a target
    return reward

# Custom environment wrapper to integrate reward_calculate
class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardEnv, self).__init__(env)

    def step(self, action):
        observation, _, done, info = self.env.step(action)
        next_observation = self.env.state
        forward_model_estimate = forward_model.predict(action)
        reward = calculate_similarity(forward_model_estimate, observation, method="mse")
        return observation, reward, done, info

# Create and wrap the environment
env = gym.make('CartPole-v1')  # Replace with your environment
env = CustomRewardEnv(env)
env = DummyVecEnv([lambda: env])  # Vectorized environment for Stable-Baselines3

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/rl_model/ppo")

# Load the trained model (optional)
# model = PPO.load("/home/gijs/Desktop/Thesis/Thesis_code/bone_inverse_rl/rl_model/ppo")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
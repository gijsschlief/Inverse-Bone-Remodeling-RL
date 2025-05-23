import numpy as np
from rl.environment import BoneEnv
from surrogate_model.cnn_surrogate import SurrogateModel

def test_env_step_returns_valid_output():
    model = SurrogateModel()
    env = BoneEnv(model)
    obs = env.reset()
    action = np.random.rand(*env.action_space.shape)
    next_obs, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert obs.shape == next_obs.shape
import numpy as np
import pickle
import torch
from docopt import docopt
from model import ActorCriticModel
from utils import create_env
import gymnasium as gym

def get_action_space_shape(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return (action_space.n,)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return tuple(action_space.nvec)
    else:
        raise NotImplementedError("Action space type not supported")

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/minigrid.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Inference device
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    # Instantiate environment
    env = create_env(config["environment"], render=True)

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, get_action_space_shape(env.action_space))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Run and render episode
    done = False
    episode_rewards = []

    # Init recurrent cell
    hxs, cxs = model.init_recurrent_cell_states(1, device)
    if config["recurrence"]["layer_type"] == "gru":
        recurrent_cell = hxs
    elif config["recurrence"]["layer_type"] == "lstm":
        recurrent_cell = (hxs, cxs)

    obs = env.reset()
    while not done:
        # Render environment
        env.render()
        # Forward model
        policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)), recurrent_cell, device, 1)
        # Sample action
        action = []
        for action_branch in policy:
            action.append(action_branch.sample().item())
        # Step environment
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
    
    # After done, render last state
    env.render()

    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))

    env.close()

if __name__ == "__main__":
    main()
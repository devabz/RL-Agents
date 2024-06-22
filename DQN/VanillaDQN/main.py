import argparse
import gym
from src import *

parser = argparse.ArgumentParser(description="Example script using argparse")

parser.add_argument(
    'env',
    type=str,
    help="Name of OpenAI Gym environment",
)
parser.add_argument(
    '-o', '--output-dir',
    type=str,
    help="Path to an output directory, is created if it doesn't exist"
)

parser.add_argument(
    '-n', '--num-episodes',
    type=int,
    help="The number episodes which the experiment will run",
    default=10,
)

parser.add_argument(
    '--epochs',
    type=int,
    help="The number of epochs for each update to the target and value networks",
    default=1,
)

parser.add_argument(
    '--record-n',
    type=int,
    help="The number of episodes to record during training",
    default=20,
)

parser.add_argument(
    '--fps',
    type=int,
    help="Frames per second",
    default=20,
)

parser.add_argument(
    '--pad-frames',
    type=int,
    help="Frames to pad after each episode",
    default=20,
)

parser.add_argument(
    '-m', '--mem-size',
    type=int,
    help="Size of experience replay memory",
    default=20,
)


parser.add_argument(
    '-b', '--batch-size',
    type=int,
    help="Batch size during training of target and value networks",
    default=20,
)


args = parser.parse_args()

env = gym.make(args.env, render_mode='rgb_array')
if not isinstance(env.action_space, gym.spaces.Discrete):
    raise 
agent = DQN(env.action_space.n, memory_size=args.mem_size)
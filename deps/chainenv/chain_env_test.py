import chain_env
import gym
from PIL import Image

env = gym.make("ChainEnv11-v1")
env.reset()

for x in range(10):
  frame = env.render(mode='rgb_array')
  im = Image.fromarray(frame)
  im.save(rf"/tmp/{x}.png")
  print(env.step(1))
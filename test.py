import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=1000)

obs, info = env.reset(seed=42)

print("hello")
print(obs)
print(info)

img = env.render()
# plt.imshow(img)
# plt.show()
print(img.shape)
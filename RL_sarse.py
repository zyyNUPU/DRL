import time
import numpy as np
import gym # type: ignore
class SarsaAgent(object):
    def __init__(self, obs_n, act_n, e_greed=0.1, gamma=0.9, learning_rate=0.01):
        self.act_n = act_n#动作空间的大小。
        self.Q = np.zeros((obs_n, act_n))#Q表，用于存储状态-动作值。
        self.lr = learning_rate
        self.gamma = gamma#折扣因子。
        self.epsilon = e_greed#epsilon贪心策略的参数。
    
    def sample(self, obs):
        if np.random.uniform(0,1) > (1.0 - self.epsilon):
            action = np.random.choice(self.act_n)#从0到self.act_n-1中随机选择一个数
        else:
            action = self.predict(obs)
        return action
    
    def predict(self, obs):#当前Q表选择Q值最高的动作
        Q_list = self.Q[obs, :]
        Qmax = np.max(Q_list)
        action_list = np.where(Q_list == Qmax)[0]#得到所有最大值的index
        action = np.random.choice(action_list)
        return action
    
    def learn(self, obs, action, next_obs, next_action, reward, done):
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)
 
def run_episode(env, agent, render=False):
    total_reward = 0
    total_steps = 0
    obs = env.reset()
    obs = obs[0]
    action = agent.sample(obs)#返回的是action的index。
    
    while True:
        next_obs, reward, done, _ , _ = env.step(action)
        next_action = agent.sample(next_obs)
        # print(type(action))
        # print(action)
        # print(type(obs))
        # print(obs)
        agent.learn(obs, action, next_obs, next_action, reward, done)
        total_reward += reward
        total_steps += 1
        obs = next_obs
        action = next_action
        if render:
            env.render()
        if done:
            break
    return total_reward, total_steps
 
def test_episode(env, agent):
    total_reward = 0
    # total_steps = 0
    obs = env.reset()
    obs = obs[0]
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, _ , _ = env.step(action)
        next_action = agent.predict(next_obs)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            break
    return total_reward
 
env = gym.make('CliffWalking-v0',render_mode="ansi")
# result = env.reset()
# print(type(result))
# print(result)
agent = SarsaAgent(obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)
for episode in range(500):
    ep_reward, ep_step = run_episode(env, agent, False)
    print("episode:{}, step:{}, reward:{}".format(episode, ep_step, ep_reward))
 
test_reward = test_episode(env, agent)
print("test_reward:{}".format(test_reward))




# from gym import envs
# env_list = envs.registry.keys()
# env_ids = [env_item for env_item in env_list]
# print('There are {0} envs in gym'.format(len(env_ids)))
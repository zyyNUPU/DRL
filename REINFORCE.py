import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import rl_utils

class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNet,self).__init__()
        self.fc1=torch.nn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        #由于损失函数是有关价值的函数，所以神经网络仍然是状态-价值的映射，而我们现在是进行策略梯度更新，需要输出动作的概率分布
        #所以需要使用softmax，即用价值的归一化来替代动作的概率分布，也就是价值越高的动作其概率也应越大
        #softmax函数将向量的元素归一化为概率分布，即计算每个动作的概率，概率之和为1，便于学习
        return F.softmax(self.fc2(x),dim=1)
    
class REINFORCE:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,device):
        self.policy_net=PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer=torch.optim.Adam(self.policy_net.parameters(),lr=learning_rate)  #使用Adam优化器
        self.gamma=gamma
        self.device=device
    
    def take_action(self,state):   #根据动作概率分布随机采样
        state=torch.tensor([state],dtype=torch.float).to(self.device)
        #probs是动作概率分布
        probs=self.policy_net(state)
        #根据动作概率分布离散化动作空间
        action_dist=torch.distributions.Categorical(probs)
        #采样,根据传进来的概率进行抽样
        action=action_dist.sample()
        return action.item()
    
    def update(self,transition_dict):
        reward_list=transition_dict['rewards']
        state_list=transition_dict['states']
        action_list=transition_dict['actions']
        
        G=0
        #显式地将梯度置0
        #模型参数，是每加一个动作更新一下
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):    #从最后一步算起，index也是如此
            reward=reward_list[i]
            state=torch.tensor([state_list[i]],dtype=torch.float).to(self.device)
            action=torch.tensor([action_list[i]]).view(-1,1).to(self.device)
            #神经网络输出的是动作的概率分布，这里是对相应动作取对数，也就是log（π（θ））
            log_prob=torch.log(self.policy_net(state).gather(1,action))
            #利用蒙特卡洛采样法计算每个时刻t往后的回报G，所以前面循环要翻转
            G=self.gamma*G+reward
            #我们知道一般的梯度下降法是对损失函数求梯度，往最大梯度的负方向进行更新参数的，目的是最小化损失函数
            #而我们这里，损失函数是累计奖励的函数，我们希望将其最大化，而不是最小化，所以这里损失函数应该加上负号
            loss=-log_prob*G  #每一步的损失函数
            loss.backward()  #反向传播计算梯度
        self.optimizer.step()   #梯度下降

learning_rate=1e-3
num_episodes=1000
hidden_dim=128
gamma=0.98
device=torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
 
env_name="CartPole-v1"
env=gym.make(env_name)
#确保环境中的随机性部分（如初始状态的选择、随机动作的结果等）是可重复的。这对于调试和结果复现非常重要。
env.reset(seed=0)
#设置 PyTorch 中所有随机数生成器的种子为 0。这样可以确保每次运行代码时，
#任何依赖随机性的操作（如权重初始化、随机抽样等）都会产生相同的结果。
torch.manual_seed(0)
#获取环境 env 的状态空间维度。env.observation_space 描述了环境中状态的特征空间。
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n
agent=REINFORCE(state_dim,hidden_dim,action_dim,learning_rate,gamma,device)
 
return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state,_ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ ,_= env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()
 
# mv_return = rl_utils.moving_average(return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('REINFORCE on {}'.format(env_name))
# plt.show()

# import torch
# import torch.nn as nn
# import torch.distributions

# class PolicyNet(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(PolicyNet, self).__init__()
#         self.fc = nn.Linear(state_size, action_size)

#     def forward(self, x):
#         x = self.fc(x)
#         return torch.softmax(x, dim=-1)  # 输出动作概率分布

# # 假设状态维度是4，动作维度是2
# state_size = 4
# action_size = 2
# policy_net = PolicyNet(state_size, action_size)

# # 假设当前状态
# state = torch.tensor([1.0, 0.5, -0.3, 0.8])

# # 获取动作概率分布
# probs = policy_net(state)

# # 创建动作分布对象
# action_dist = torch.distributions.Categorical(probs)

# # 从分布中采样一个动作
# action = action_dist.sample()

# print("动作概率分布:", probs)
# print("采样的动作:", action.item())

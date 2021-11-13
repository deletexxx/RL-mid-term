from typing import (
    Tuple,
)

import torch
import numpy as np
import random

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

class Tree(object):
    data_pointer = 0
    
    #存经验的下标和对应的优先级在树中
    def __init__(self,channels,capacity,device):
        self.device = device
        self.capacity = capacity  #capacity是叶节点个数
        self.tree = np.zeros(2 * capacity -1)  
        #有capacity-1个父节点, capacity个子节点存优先级
        #self.data = np.zeros(capacity+1,dtype = object) #存叶节点对应的数据data[叶子节点编号id]=data，在本实验是经验的下标
        self.__m_states = torch.zeros((capacity,channels,84,84),dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        
    def add(self, p ,state,action,reward,done):
        idx = self.data_pointer + self.capacity - 1
        
        #self.data[self.data_pointer] = data  #增加一个记录
        self.__m_states[self.data_pointer] = state
        self.__m_actions[self.data_pointer] = action
        self.__m_rewards[self.data_pointer] = reward
        self.__m_dones[self.data_pointer] = done
        self.updatetree(idx,p)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  #已经超出范围
            self.data_pointer = 0
            
    def updatetree(self,idx,p):
        change = p - self.tree[idx]  #改变位置
        
        self.tree[idx] = p   #将对应位置的叶节点存的值改为p
        while idx != 0:   #这样比递归更快
            idx = (idx - 1) // 2
            self.tree[idx] += change       
    
    #v是分好第i段的均匀采样值,从树里找对应的叶子节点拿样本数据、样本叶子节点序号和样本优先级
    def get_leaf(self,v):  
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  #没有子节点了
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]: #比左孩子的值还小
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx] #在右孩子继续找
                    parent_idx = cr_idx
    
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], \
            self.__m_states[data_idx, :4].to(self.device).float(), \
            self.__m_states[data_idx, 1:].to(self.device).float(), \
            self.__m_actions[data_idx].to(self.device), \
            self.__m_rewards[data_idx].to(self.device).float(), \
            self.__m_dones[data_idx].to(self.device)
    
    @property
    def total_p(self):
        return self.tree[0] #返回根节点
    
class ReplayMemory(object):   
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.entries = 0
        '''
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0
        '''
        self.tree = Tree(channels,capacity,device)   #初始化一棵树
        
    #经验存储
    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        
        self.entries += 1
        p = np.max(self.tree.tree[-self.tree.capacity:])
        if p == 0:
            p = self.abs_err_upper    
        
        self.tree.add(p, folded_state, action, reward, done)  #在树里存入数据


    #从 Replay Memory中采样
    #优先级检验回放：按照TD-error对经验进行排序，越大表示待提升空间越大，被拿出来训练的优先级越高
    def sample(self, batch_size: int):
        idxs = np.empty((batch_size,),dtype = np.int32)
        ISweight = np.empty((batch_size,1))
        
        b_state = []
        b_next = []
        b_action = []
        b_reward = []
        b_done = []
        
        pri_seg = self.tree.total_p / batch_size #把prority分成batch_size个部分，每个部分大小为batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        if min_prob == 0:
            min_prob = 0.00001
        
        for i in range(batch_size):
            start = pri_seg * i
            end = pri_seg * (i+1)
            v = np.random.uniform(start,end) #在start和end之间均匀采样一个
            idx, p, state, next, action, reward, done = self.tree.get_leaf(v) #得到相应的优先级和下标
            
            idxs[i] = idx
            P = p / self.tree.total_p  #为了求w
            ISweight[i,0] = np.power(P / min_prob,-self.beta)
            
            b_state.append(state)
            b_next.append(next)
            b_action.append(action)
            b_reward.append(reward)
            b_done.append(done)
        
        b_state = torch.stack(b_state)
        b_next = torch.stack(b_next)
        b_action = torch.stack(b_action)
        b_reward = torch.stack(b_reward)
        b_done = torch.stack(b_done)
        
        batch = (b_state,b_next,b_action,b_reward,b_done)
        
        return idxs,batch,ISweight
        
    
    def __len__(self) -> int:
        return self.entries
    
    def update(self,indices,p):
        p += self.epsilon #变种一
        clipped_error = np.minimum(p.cpu().data.numpy(),self.abs_err_upper)
        ps = np.power(clipped_error,self.alpha)
        for idx,valueP in zip(indices,ps):
            self.tree.updatetree(idx,valueP)

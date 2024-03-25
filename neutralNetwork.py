"""
《 Python神经网络编程 》的代码实现
实现一个三层的神经网络
使用 Sigmoid 函数作为激发函数，在代码中为 scipy.special.expit()
"""

import tool
import numpy as np
import scipy.special as ss

class neutralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate,Wih=None,Who=None):
        self.inode = inputnodes
        self.hnode = hiddennodes
        self.onode = outputnodes
        self.lr = learningrate
        if (Wih):
            self.Wih = np.load('weight/Wih.npy')
        else:
            self.Wih = np.random.rand(self.hnode, self.inode) - 0.5
        if (Who):
            self.Who = np.load('weight/Who.npy')
        else:
            self.Who = np.random.rand(self.onode,self.hnode) - 0.5
        self.activation = lambda x:ss.expit(x)

    def train(self,input_list,target_list):
        """
        step 1 计算输出
        step 2 反向传播误差，优化链接权重
        输入列表的元素值应当与输入节点数量一致
        """
        # step 1
        input = np.array(input_list,ndmin=2).T
        target = np.array(target_list,ndmin=2).T
        hidden_input = np.dot(self.Wih,input)
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(self.Who,hidden_output)
        final_output = self.activation(final_input)
        # step 2
        output_error = target - final_output
        hidden_error = np.dot(self.Who.T,output_error)
        # 使用梯度下降更新参数
        self.Who += self.lr * np.dot((output_error * final_output * (1.0 - final_output)),np.transpose(hidden_output))
        self.Wih += self.lr * np.dot((hidden_error * hidden_output * (1.0 - hidden_output)),np.transpose(input))

    def query(self,input_list):
        """
        用于观察神经网络训练情况
        提供输入数据，返回输出数据
        """
        input = np.array(input_list,ndmin = 2).T
        hidden_input = np.dot(self.Wih,input)
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(self.Who,hidden_output)
        final_output = self.activation(final_input)
        return final_output

    def saveWeight(self):
        np.save('weight/Wih',self.Wih)
        np.save('weight/Who',self.Who)


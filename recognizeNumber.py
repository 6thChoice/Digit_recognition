import numpy as np
import neutralNetwork as nn

class recognizeNumber(nn.neutralNetwork):
    def __init__(self,inputnode,hiddennode,outputnode,learningrate,Wih=None,Who=None):
        super(recognizeNumber,self).__init__(inputnode,hiddennode,outputnode,learningrate,Wih=Wih,Who=Who)
        self.inode = inputnode
        self.hnode = hiddennode
        self.onode = outputnode
        self.lr = learningrate

    def whichIsBiggest(li):
        i = 0
        max = 0
        for ind in range(len(li)):
            if li[ind] > max:
                i = ind
                max = li[ind]
        return i

    def train(self,train_file_location,ehcos = 1):
        train_data_file = open(train_file_location,'r')
        train_data_list = train_data_file.readlines()
        train_data_file.close()

        for i in range(ehcos):
            for line in train_data_list:
                all_value = line.split(',')
                input = ((np.asfarray(all_value[1:]) / 255.0 * 0.99) + 0.01)
                target = np.zeros(self.onode)
                target[int(all_value[0])] = 0.99
                super(recognizeNumber,self).train(input,target)
        super().saveWeight()

    def query(self,test_file_location):
        test_data_file = open(test_file_location,'r')
        test_data_list = test_data_file.readlines()
        test_data_file.close()

        count = 0
        max_len = len(test_data_list)

        for line in test_data_list:
            all_value = line.split(',')
            input = ((np.asfarray(all_value[1:]) / 255.0 * 0.99) + 0.01)
            res_li = super(recognizeNumber,self).query(input)
            result = recognizeNumber.whichIsBiggest(res_li)
            print("神经网络输出值为：", result, "，实际值为：", all_value[0])
            if result == int(all_value[0]):
                count += 1
        print("测试数为：", max_len)
        print("拟合率为：", count / max_len * 100, "%")
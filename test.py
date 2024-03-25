import recognizeNumber as rn
import tool
import numpy as np

inputnode = 784
hiddennode = 200
outputnode = 10
learningrate = 0.2

"""rw = rn.recognizeNumber(inputnode,hiddennode,outputnode,learningrate)
rw.train('doc/mnist_train.csv')
rw.query('doc/mnist_test.csv')"""

rw2 = rn.recognizeNumber(inputnode,hiddennode,outputnode,learningrate)
rw2.train('doc/mnist_train.csv',4)
rw2.query('doc/mnist_test.csv')

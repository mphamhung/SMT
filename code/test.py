from preprocess import *
from lm_train import *
from log_prob import *

import os
s = 'That process is underway'

data_dir = os.getcwd() + '/../data/'
# for data in os.listdir(data_dir):
#     with open(data_dir+data) as f:
#         for sent in f.readlines():
#             print(sent)
#             print(preprocess(sent, 'f'))

# lm_train(data_dir, 'e', 'test')
# lm_train(data_dir, 'f', 'test')
# print(preprocess(s, 'e'))
if os.path.getsize(os.getcwd()+'/etest.pickle') > 0:      
    with open(os.getcwd()+'/etest.pickle', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        lm1 = unpickler.load()


log_prob(s, lm1)

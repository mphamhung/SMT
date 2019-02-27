from preprocess import *
from lm_train import *

import os
s = 'That process is underway No. Col. '

data_dir = os.getcwd() + '/../data/'
# for data in os.listdir(data_dir):
#     with open(data_dir+data) as f:
#         for sent in f.readlines():
#             print(sent)
#             print(preprocess(sent, 'f'))

lm_train(data_dir, 'e', 'test')
lm_train(data_dir, 'f', 'test')
# print(preprocess(s, 'e'))

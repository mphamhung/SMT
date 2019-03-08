from preprocess import *
from lm_train import *
from log_prob import *
from perplexity import *

import os
s = 'That process is underway'

# data_dir = os.getcwd() + '/../data/'
# # for data in os.listdir(data_dir):
# #     with open(data_dir+data) as f:
# #         for sent in f.readlines():
# #             print(sent)
# #             print(preprocess(sent, 'f'))

# # lm_train(data_dir, 'e', 'test')
# # lm_train(data_dir, 'f', 'test')
# # print(preprocess(s, 'e'))
if os.path.getsize(os.getcwd()+'/etest.pickle') > 0:      
    with open(os.getcwd()+'/etest.pickle', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        lm1 = unpickler.load()
if os.path.getsize(os.getcwd()+'/ftest.pickle') > 0:
    with open(os.getcwd()+'/ftest.pickle', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        lm2 = unpickler.load()


log_prob(s, lm1, smoothing = True, delta = 0.1, vocabSize = 400)

f = open("Task3.txt", "w+")

test_dir = '/u/cs401/A2_SMT/data/Hansard/Testing/'
f.write('English Testing')
f.write(f'No delta: { preplexity(lm1, test_dir, "e", smoothing = False, delta = 0)}')
f.write('delta 0.1: ', preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.1))
f.write('delta 0.25: ',preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.25))
f.write('delta 0.5: ',preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.5))
f.write('delta 0.75: ',preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.75))

f.write('French Testing')
f.write('No delta: ', preplexity(lm2, test_dir, 'f', smoothing = False, delta = 0))
f.write('delta 0.1: ',preplexity(lm2, test_dir, 'f', smoothing = True, delta = 0.1))
f.write('delta 0.25: ',preplexity(lm2, test_dir, 'f', smoothing = True, delta = 0.25))
f.write('delta 0.5: ',preplexity(lm2, test_dir, 'f', smoothing = True, delta = 0.5))
f.write('delta 0.75: ',preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.75))

#print('-----Testing English Perplexity-----')
#print('No delta: ', preplexity(lm1, test_dir, 'e', False, 0))
#print('Delta 0.05: ', preplexity(lm1, test_dir, 'e', True, 0.05))
#print('Delta 0.1: ', preplexity(lm1, test_dir, 'e', True, 0.10))
#print('Delta 0.25: ', preplexity(lm1, test_dir, 'e', True, 0.25))
#print('Delta 0.5: ', preplexity(lm1, test_dir, 'e', True, 0.5))
#print('Delta 0.75: ', preplexity(lm1, test_dir, 'e', True, 0.75))
#print('Delta 0.85: ', preplexity(lm1, test_dir, 'e', True, 0.85))
#print('Delta 0.95: ', preplexity(lm1, test_dir, 'e', True, 0.95))
#print('Delta 1: ', preplexity(lm1, test_dir, 'e', True, 1.0))

print('-----Testing French Perplexity-----')
print('No delta: ', preplexity(lm2, test_dir, 'f', False, 0))
print('Delta 0.05: ', preplexity(lm2, test_dir, 'f', True, 0.05))
print('Delta 0.1: ', preplexity(lm2, test_dir, 'f', True, 0.10))
print('Delta 0.25: ', preplexity(lm2, test_dir, 'f', True, 0.25))
print('Delta 0.5: ', preplexity(lm2, test_dir, 'f', True, 0.5))
print('Delta 0.75: ', preplexity(lm2, test_dir, 'f', True, 0.75))
print('Delta 0.85: ', preplexity(lm2, test_dir, 'f', True, 0.85))
print('Delta 0.95: ', preplexity(lm2, test_dir, 'f', True, 0.95))
print('Delta 1: ', preplexity(lm2, test_dir, 'f', True, 1.0))

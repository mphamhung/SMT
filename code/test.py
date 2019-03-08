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
if os.path.getsize(os.getcwd()+'/eEvalLM.pickle') > 0:      
    with open(os.getcwd()+'/eEvalLM.pickle', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        lm1 = unpickler.load()
if os.path.getsize(os.getcwd()+'/fEvalLM.pickle') > 0:
    with open(os.getcwd()+'/fEvalLM.pickle', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        lm2 = unpickler.load()

e_sent = 'It is indeed a great honour to be entrusted with this task.'
f_sent = 'Chers collegues, vous me faites un grand honneur en me confiant cette tache.'

deltas = [0,0.1,0.25,0.5,0.75]

lptxt = e_sent + '\n\n'
for d in deltas:
    l = log_prob(e_sent, lm1, smoothing = True, delta = d, vocabSize = 400)
    lptxt += f'Delta = {d}' + f'log probability = {l} \n'

lptxt += f_sent + '\n\n'
for d in deltas:
    l = log_prob(f_sent, lm2, smoothing = True, delta = d, vocabSize = 400)
    lptxt += f'Delta = {d}' + f'log probability = {l}\n'
f = open("Task3.txt", "w+")

f.write('---- Log probabilities with a sentence ----\n\n')
f.write(lptxt)
discussion = '''
Discussion:

We notice that the MLE model has the lowest perplexity and as we increase the delta smoothing, the perplexity also increases. Mathematically, we notice that the log likelihood decreases as delta increases, and the perplixity is inversly proportional to the log likelihood, thus explaining the increases in perplexity. From the below results we can conclude that increasing delta towards 1 results in a worse performing model. 
Refering to the test and training corpus, they are both from the Canadian Hansards, so their is potential for a large overlap in language. Adding delta smoothing to unknown words will thus not help that much.

---------
'''

f.write(discussion)

test_dir = '/u/cs401/A2_SMT/data/Hansard/Testing/'
f.write('English Perplexity Testing')
f.write('\n\n')
f.write('No delta: ' + str(preplexity(lm1, test_dir, 'e', smoothing = False, delta = 0)))
f.write('\n')
f.write('delta 0.1: ' + str(preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.1)))
f.write('\n')
f.write('delta 0.25: ' + str(preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.25)))
f.write('\n')
f.write('delta 0.5: ' + str(preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.5)))
f.write('\n')
f.write('delta 0.75: ' + str(preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.75)))
f.write('\n')
f.write('\n')
f.write('French Perplexity Testing')
f.write('\n')
f.write('\n')
f.write('No delta: ' + str(preplexity(lm2, test_dir, 'f', smoothing = False, delta = 0)))
f.write('\n')
f.write('delta 0.1: ' + str(preplexity(lm2, test_dir, 'f', smoothing = True, delta = 0.1)))
f.write('\n')
f.write('delta 0.25: ' + str(preplexity(lm2, test_dir, 'f', smoothing = True, delta = 0.25)))
f.write('\n')
f.write('delta 0.5: ' + str(preplexity(lm2, test_dir, 'f', smoothing = True, delta = 0.5)))
f.write('\n')
f.write('delta 0.75: ' + str(preplexity(lm1, test_dir, 'e', smoothing = True, delta = 0.75)))
f.write('\n')

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

# print('-----Testing French Perplexity-----')
# print('No delta: ', preplexity(lm2, test_dir, 'f', False, 0))
# print('Delta 0.05: ', preplexity(lm2, test_dir, 'f', True, 0.05))
# print('Delta 0.1: ', preplexity(lm2, test_dir, 'f', True, 0.10))
# print('Delta 0.25: ', preplexity(lm2, test_dir, 'f', True, 0.25))
# print('Delta 0.5: ', preplexity(lm2, test_dir, 'f', True, 0.5))
# print('Delta 0.75: ', preplexity(lm2, test_dir, 'f', True, 0.75))
# print('Delta 0.85: ', preplexity(lm2, test_dir, 'f', True, 0.85))
# print('Delta 0.95: ', preplexity(lm2, test_dir, 'f', True, 0.95))
# print('Delta 1: ', preplexity(lm2, test_dir, 'f', True, 1.0))

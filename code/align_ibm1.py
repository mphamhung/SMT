from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    eng, fre = read_hansard(train_dir, num_sentences)
        
    # Initialize AM uniformly
    t = initalize(eng,fre)
    
    # Iterate between E and M steps
    for _ in range(max_iters):
        t = em_step(t,eng,fre)
    
    AM = t
    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return AM
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""

    sentCounter = 0

    e_sents = []
    f_sents = []

    for ffile in os.listdir(train_dir):
        if ffile[-1] == 'e':
            engfile = ffile
            frefile = ffile[:-1] + 'f'
            try:
                edoc = open(train_dir+engfile)
                fdoc = open(train_dir+frefile)
                e = edoc.readlines()
                f = fdoc.readlines()
                edoc.close()
                fdoc.close()
            except FileNotFoundError:
                print('not found')
                continue

            if len(e) <= num_sentences - sentCounter:
                e_sents += [preprocess(sent, 'e') for sent in e]
                f_sents += [preprocess(sent, 'f') for sent in f]
                sentCounter += len(e)
            else:
                e_sents += [preprocess(sent, 'e') for sent in e[:num_sentences-sentCounter]]
                f_sents += [preprocess(sent, 'f') for sent in f[:num_sentences-sentCounter]]
                break

    return e_sents,f_sents

def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	AM = {}

    for i in range(len(eng)):
        for e_word in eng[i].split(' '):
            if e_word not in AM.keys():
                AM[e_word] = {}

            for f_word in fre[i].split(' '):
                AM[e_word][f_word] = 1

    for e_word in AM.keys():
        for f_word in AM[e_word].keys():
            AM[e_word][f_word] = float(1/len(AM[e_word].keys()))

    return AM
    
def em_step(t, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""

    tcount = {}
    total = {}
    for j in range(len(eng)):
        E = eng[j].split(' ')
        F = fre[j].split(' ')
        for f in set(F):
            denom_c = 0
            for e in set(E):
                denom_c += t[e][f]*F.count(f)
            for e in set(E):
                if e not in tcount.keys():
                    tcount[e] = {}
                    total[e] = 0
                if f not in tcount[e].keys():
                    tcount[e][f] = 0

                tcount[e][f] += (t[e][f]*F.count(f)*E.count(e))/denom_c
                total[e] += (t[e][f]*F.count(f)*E.count(e))/denom_c    
    for e in total.keys():
        for f in tcount[e]:
            t[e][f] = tcount[e][f]/total[e]

    return t
        

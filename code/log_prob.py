from preprocess import *
from lm_train import *
from math import log2

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
	"""
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""

	#TODO: Implement by student.
	splitted = sentence.split(' ')
	log_prob = 0
	for t in range(len(splitted)-1):
		wt = splitted[t]
		wt_1 = splitted[t+1]

		count_wt = 0.0
		count_wt_1 = 0.0
		if wt in LM['uni'].keys():
			count_wt += LM['uni'][wt]
			if wt_1 in LM['bi'][wt].keys():
				count_wt_1 += LM['bi'][wt][wt_1]


		if count_wt and count_wt_1:
			if smoothing:
				log_prob += log2((count_wt_1+delta)/(count_wt+delta*vocabSize))
			else:
				log_prob += log2((count_wt_1/count_wt))
		else:
			log_prob += float('-inf')

	#	print(log_prob)
	return log_prob        

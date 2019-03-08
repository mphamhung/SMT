from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
	"""
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM

	INPUTS:

	data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained

	OUTPUT

	LM			: (dictionary) a specialized language model

	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts

	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
			LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
	"""
	LM = {}
	LM['uni'] = {}
	LM['bi'] = {}

	for data in os.listdir(data_dir):
		if data[-1] != language:
			continue
		with open(data_dir + '/' + data) as f:
			for sent in f.readlines():
				processed_sent = preprocess(sent,language)
				for word in processed_sent.split(' '):
					if word in LM['uni'].keys():
						LM['uni'][word] += 1
					else:
						LM['uni'][word] = 1

				for i in range(len(processed_sent.split(' '))-1):
					w1 = processed_sent.split(' ')[i]
					w2 = processed_sent.split(' ')[i+1]
					if w1 in LM['bi'].keys():
						if w2 in LM['bi'][w1].keys():
							LM['bi'][w1][w2] += 1
						else:
							LM['bi'][w1][w2] = 1
					else:
						LM['bi'][w1] = {}
						LM['bi'][w1][w2] = 1
				if processed_sent.split(' ')[-1] not in LM['bi'].keys():
					LM['bi'][processed_sent.split(' ')[-1]] = {}
	language_model = LM
	#Save Model
	with open(fn_LM+'.pickle', 'wb') as handle:
		pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return language_model

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle
import os

import decode
from align_ibm1 import *
from BLEU_score import *
from lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model 
    """
    if use_cached:
        try:
            path = os.getcwd() +'/' + fn_LM +'.pickle'
            with open(path, 'rb') as f:
                LM = pickle.load(path)
        except FileNotFoundError:
            LM = lm_train(data_dir,language,fn_LM)
    else:
        LM = lm_train(data_dir,language,fn_LM)
    
    return LM

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data 
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model 
    """
    if use_cached:
        try:
            path = os.getcwd() +'/' + fn_AM +'.pickle'
            with open(path, 'rb') as f:
                AM = pickle.load(path)
        except FileNotFoundError:
            AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)

    return AM


def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """

    return [BLEU_score(eng_decoded[i], [eng[i],google_refs[i]], n) for i in range(len(eng))]
   

def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    data_dir = str(args.data_dir)
    max_iter = int(args.max_iters)
    use_cached = bool(args.use_cached)
    
    AM_names = {'1k': 1000, '10k': 10000, '15k': 15000, '30k': 30000}
    
    AMs = {name: _getAM(data_dir, num, max_iter, name+'AM', use_cached) for name, num in AM_names.items()}
    
    
    LM = {'e': _getLM(data_dir,'e' ,'eEvalLM', use_cached), 'f': _getLM(data_dir, 'f', 'fEvalLM', use_cached)}


    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    '''
    f = open("Task5.txt", 'w+')
    f.write(discussion) 
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i, AM in enumerate(AMs):
        
        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(...)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    '''
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    parser.add_argument("--data_dir", help="data directory", required = True)
    parser.add_argument("--max_iters", help = "The maximum number of iterations for EM", default = 100)
    parser.add_argument("--use_cached", help = "bool to determine cached use", default = True)

    args = parser.parse_args()


    main(args)

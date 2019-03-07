import math


def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of
    reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on

    DO NOT concatenate the measurments. N=2 means only bigram. Do not average
    /incorporate the uni-gram scores.

    INPUTS:
    sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
    references:	(list) List containing reference sentences. ["SENTSTART je suis
    faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
    n :			(int) one of 1,2,3. N-Gram level.


    OUTPUT:
    bleu_score :	(float) The BLEU score
    """
    # TODO: Implement by student

    R = []
    for sent in references:
        R += sent.split(' ')

    R = set(R)

    #p1
    bleu_score = len([ canditate_word for canditate_word in canditate.split(' ') if i in R])/float(len(canditate.split(' ')))

    if n >= 2:
        #p2
        bleu_score *= len([k for k in get_ngrams([canditate],2) if k in get_ngrams(references,2,True)])/float(len(get_ngrams([canditate],2)))

    if n >= 3:
        #p3
        bleu_score *= len([k for k in get_ngrams([canditate],3) if k in get_ngrams(references,3,True)])/float(len(get_ngrams([canditate],3)))

    bleu_score **= 1/n

    if brevity:
        c_i = len(candidate.split(' '))
        lengths = [abs(c_i - len(ref.split(' '))) for ref in references]
        best_ind = lengths.index(min(lengths))
        r_i = float(len(references[best_ind]))

	
        BP = math.exp(1-(r_i/c_i)) if r_i>=c_i else 1

        bleu_score *= BP

    return bleu_score

def get_ngrams(listOfSents,n, isSet = False):
    ngrams = []
    for sents in listOfSents:
        tokens = sents.split(' ')
        for i in range(n,len(tokens)):
            ngrams.append(' '.join(tokens[i-n:i]))
    if isSet:
        ngrams = set(ngrams)
    return bigrams



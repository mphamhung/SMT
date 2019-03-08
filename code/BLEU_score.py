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
    print(f"CANDIDATE: {candidate}")
    print(f"REFS: {references}")
    print(f"n = {n}")

    R = []
    for sent in references:
        R += sent.split(' ')

    R = set(R)

    #p1
    bleu_score = len([ candidate_word for candidate_word in candidate.split(' ') if candidate_word in R])/float(len(candidate.split(' ')))

    if n >= 2:
    #    #p2
        c_bigrams = get_ngrams([candidate],2)
        r_bigrams = get_ngrams(references,2)
        bleu_score *= (len([k for k in c_bigrams if k in r_bigrams]))/float(len(c_bigrams))
        #bleu_score *= len([k for k in get_ngrams([candidate],2)/ if k in get_ngrams(references,2,True)])/float(len(get_ngrams([candidate],2)))

    if n >= 3:
        #p3
        c_trigrams = get_ngrams([candidate],3)
        r_trigrams = get_ngrams(references, 3)
        bleu_score *= (len([k for k in c_trigrams if k in r_trigrams]))/float(len(c_trigrams))
        #bleu_score *= len([k for k in get_ngrams([candidate],3) if k in get_ngrams(references,3,True)])/float(len(get_ngrams([candidate],3)))

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
        for i in range(n,len(tokens)+1):
            ngrams.append(' '.join(tokens[i-n:i]))
    #print(ngrams) 
    return ngrams



import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function
    out_sentence = in_sentence      

    out_sentence = re.sub(r' *\n', r'', out_sentence) #remove \n
    out_sentence = re.sub(r'([,!?:\);])', r' \1', out_sentence) #remove EOS not periods
    out_sentence = re.sub(r'(\()', r'\1 ', out_sentence) #add space after opeing (
    out_sentence = re.sub(r'(\w+)(\.)$', r'\1 \2', out_sentence) #EOS periods
    
    while re.search(r'(\(.+[A-z])(\-)([A-z].+\))', out_sentence):
        out_sentence = re.sub(r'(\(.+[A-z])(\-)([A-z].+\))', r'\1 \2 \3', out_sentence) #dashes between parentheses

    while re.search(r'(\d)([\+=<>\-])(\d)', out_sentence):
        out_sentence = re.sub(r'(\d+)(\-)(\d+)', r'\1 \2 \3', out_sentence) #minus signs
        out_sentence = re.sub(r'(\d)([\+=<>])(\d)', r'\1 \2 \3', out_sentence) #other math signs
    
    if language == 'f':
        out_sentence = re.sub(r'(\b(l|j|t|qu|puisqu|lorsqu)\')', r'\1 ', out_sentence, flags=re.IGNORECASE)

    out_sentence = "SENTSTART " + out_sentence + " SENTEND"

    return out_sentence

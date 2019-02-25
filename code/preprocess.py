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
	out_sentence = re.sub(r'([.!?,:;()\-+<>="]+)', r' \1', in_sentence)
    
    if language == 'f':
        out_sentence = re.sub(r'(\b(l|j|t|qu|puisqu|lorsqu)\')', r'\1 ', out_sentence)
    
    out_sentence = "SENTSTART " + out_sentence + " SENTEND"
    
    return out_sentence
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
    
	out_sentence = re.sub(r'([.!?,:;()"]+ )', r' \1', out_sentence) #sf punct, commas, semicolons, parentheses
    out_sentence = re.sub(r'(\(.+[A-z])(\-)([A-z].+\))', r'\1 \2 \3', out_sentence) #dashes between parentheses
    out_sentence = re.sub(r'(\d+(\-|\+|=|<|>|)\d+)')

    if language == 'f':
        out_sentence = re.sub(r'(\b(l|j|t|qu|puisqu|lorsqu)\')', r'\1 ', out_sentence)
    
    out_sentence = "SENTSTART " + out_sentence + " SENTEND"

    return out_sentence
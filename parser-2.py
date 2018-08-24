import nltk
from nltk.util import ngrams
import re
from nltk.tokenize import sent_tokenize
from nltk import load

def processSentence(sentences,posLex,negLex,tagger):
    wordnoun=[]
    posneg= posLex | negLex
    for sent in sentences:
        sent=re.sub('[^a-zA-Z\d]',' ',sent)#replace chars that are not letters or numbers with a spac
        sent=re.sub(' +',' ',sent).strip()#remove duplicate spaces
        #tokenize the sentence
        terms = nltk.word_tokenize(sent.lower())   
        POStags=['NN'] # POS tags of interest 		
        POSterms=getPOSterms(terms,POStags,tagger)
        nouns=POSterms['NN']
        firstword='not'
        #anyword=text
        wordnoun+=getnounfourgrams(firstword,posneg,terms, nouns)
    return wordnoun
        
        
def getnounfourgrams(firstword,posneg,terms, nouns):
    result=[]
    fourgrams = ngrams(terms,4) #compute 2-grams
    for pn in fourgrams:
        if pn[0]=='not'  and pn[2] in posneg and pn[3] in nouns:             
            result.append(pn)   
    return result
        
def getPOSterms(terms,POStags,tagger):
    tagged_terms=tagger.tag(terms)#do POS tagging on the tokenized sentence
    POSterms={}
    for tag in POStags:POSterms[tag]=set()
    #for each tagged term
    for pair in tagged_terms:
        for tag in POStags: # for each POS tag 
            if pair[1].startswith(tag): POSterms[tag].add(pair[0])
    return POSterms

    
def loadLexicon(fname):
    newLex=set()
    lex_conn=open(fname)
    #add every word in the file to the set
    for line in lex_conn:
        newLex.add(line.strip())# remember to strip to remove the lin-change character
    lex_conn.close()
    return newLex    
    

def run(fpath):
    _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
    tagger = load(_POS_TAGGER)
    f=open(fpath)
    text=f.read().strip()  
    f.close()
    sentence=sent_tokenize(text)
    print ('NUMBER OF SENTENCES: ',len(sentence))
    posLex=loadLexicon('positive-words.txt')
    negLex=loadLexicon('negative-words.txt')
    fourword=processSentence(sentence,posLex,negLex,tagger)
    return fourword
    
def getTop3(D): 
    diclist=sorted(D, key=D.get, reverse=True)[:3]
    return(diclist)
    
if __name__=='__main__':
    print (run('input.txt'))
    D = {'a':1, 'b':50, 'c': 359,'d':11,'e':5, 'f':580}
    diclist=getTop3(D)
    print (diclist)
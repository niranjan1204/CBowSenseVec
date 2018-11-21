from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
import numpy
import gc

# Preprocessing
data1 = open('wiki_sentences.txt','r').read()
data2 = open('wsd_sentences.txt', 'w')
stopwords = set(stopwords.words('english'))

def preprocess_(wiki):
        wiki = wiki.lower()
        lemma = WordNetLemmatizer() 
        wiki_list = wiki.split("\n")
        print 'Loading Data:'
        print 'Initial Data Size : ', len(wiki_list)
	data = [word_tokenize(x) for x in wiki_list]
	return data

#WordRepresentations
def vectors(vector1):
	iter1 = 0; size = 0
	lemma = WordNetLemmatizer()
	while(iter1 < len(vector1)):
                flag = 0
		if len(vector1[iter1]) > 20 and len(vector1[iter1]) < 500:
                        vector_t = []
                	for y in range(len(vector1[iter1])):
                        	lm = lemma.lemmatize((vector1[iter1][y]).lower())
                                if(len(lm) > 3):
                                    vector_t.append(lm)
                                if lm == 'hard' or lm == 'line' or lm == 'interest' :
                                    flag = 1
                        if flag == 1:
                            for x in vector_t:
                                data2.write(x + ' ')
                            data2.write('\n')
                            size += 1
                iter1 += 1	
        print "Dataset Size : ", size 


def main():
        data = preprocess_(data1)
	print 'Execution Begins:'
        vectors(data)

if __name__ == '__main__' :
    main()

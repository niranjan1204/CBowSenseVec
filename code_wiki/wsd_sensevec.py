from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
import numpy, gc

# Preprocessing
data1 = open('wsd_sentences.txt','r').read()
stopwords = set(stopwords.words('english'))
file_words = 'words-test.txt'

def preprocess_(wiki):
        wiki_list = wiki.split("\n")
        wiki_list = [word_tokenize(x) for x in wiki_list]
        print 'Dataset Size', len(wiki_list)
        return wiki_list
        

#WordRepresentations
def vectors(vector_1):
        dictionary = [] ; vector_ = []
	temp = Word2Vec(vector_1 , min_count = 10, size = 50)
	vocab = temp.wv.vocab
        print('Word Vector Creation:')
        print 'Vocabulary Size : ', len(vocab)
	for x in vector_1:
		vec_1 = []
		for y in x:
			if y in vocab and len(y) > 2:
				vec_1.append(y)
		vector_.append(vec_1)
                	
        del vector_1
        gc.collect()

        local = numpy.zeros((len(vector_), 50))
        print('Context Vector Creation:')
	for i in range(len(vector_)):
                for j in range(len(vector_[i])):
                        local[i] += temp[vector_[i][j]]
                local[i] /= len(vector_[i]) + 1
	
        for i in range(len(vector_)):
                arr1 = []
                for j in range(len(vector_[i])):
                        arr1.append(temp[vector_[i][j]] + local[i])
                dictionary.append(arr1)
                gc.collect()
                    
	return dictionary, vector_, vocab				


def cluster(context_vec, words, vocab):
	for x in vocab:
		batch = []
		count = 0
		for i in range(len(words)):
			for j in range(len(words[i])):
				if words[i][j] == x:
					batch.append(context_vec[i][j])

		model = DBSCAN(eps = 0.001, min_samples = 50, metric = 'cosine').fit(batch) 
		labels = model.labels_
		del model		

		for i in range(len(words)):
                        for j in range(len(words[i])):
                                if words[i][j] == x:
                                        words[i][j] = words[i][j] + '_' + str(labels[count])
					count = count+1
	return words


def main():
        data = preprocess_(data1)
	print 'Execution Begins:'
        context, words, vocab = vectors(data)
        print 'Clustering :'
        x = open(file_words, 'r').read()
        x = x.split('\n')
        arr = []
        for z in x:
                if z in vocab:
                        arr.append(z)
        
        words_new = cluster(context, words, arr)
	words_final = Word2Vec(words_new, min_count = 5, size = 50)

        for x in arr:
            count = 0
            while True:
                try:
                    print words_final.most_similar_cosmul(x + '_' + str(count)), ' >>>>> ', x , '\n'
                    count += 1
                except:
                    print '\n'
                    break

if __name__ == '__main__' :
    main()

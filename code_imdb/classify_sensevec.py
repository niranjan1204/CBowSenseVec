import os, re, time, gensim, numpy, gc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Preprocessing

path1 = "./dataset/train/neg"
path2 = "./dataset/train/pos"
path3 = "./dataset/test/neg"
path4 = "./dataset/test/pos"
files1 = os.listdir(path1)
files2 = os.listdir(path2)
files3 = os.listdir(path3)
files4 = os.listdir(path4)

maxfeatures, vec_length = 5000, 50
stop_words = set(stopwords.words('english'))

def preprocess_(files, path):
	data_ = []
	for f in files:
		input_ = open(os.path.join(path,f),'r').read()
		output_ = os.path.splitext(f)[0]
		temp_, j = [], 0
		for i in range(len(output_)):
			if output_[i] == '_':
				output_ = output_[i+1:] 
				break
		input_ = re.sub(r'<br /><br />',' ', input_)
		input_ = re.sub(r'[^a-zA-Z]+',' ', input_)
		data_.append([input_, output_])
	return data_


#WordRepresentations	

def create_vectors(vector_raw):
	print 'Word_vectors:'
    	wvec1 = Word2Vec(vector_raw,  min_count = 100, size = vec_length)
        tfidf_vec1 = TfidfVectorizer(stop_words = 'english', min_df = 100,  max_features = maxfeatures)
        vector_iter1 = []
        for x in vector_raw:
            vector_iter1.append(str(x))
        tfidf1 = tfidf_vec1.fit_transform(vector_iter1)
        vocab11 = wvec1.wv.vocab
        vocab21 = tfidf_vec1.get_feature_names()
        dictionary1 = numpy.zeros((len(vocab21), vec_length))
        for x in range(len(vocab21)):
            	y = vocab21[x]
            	if y in vocab11:
                	dictionary1[x] = wvec1[y]
        dictionary1 = tfidf1.dot(dictionary1)

        print 'Clustering:'

	batch = {}; labels = {}; count = {}
	for z in vocab11:
		batch[z] = []
		count[z] = 0

	for i in range(len(vector_raw)):
		for j in range(len(vector_raw[i])):
			if vector_raw[i][j] in batch:
                                if len(batch[vector_raw[i][j]]) == 0:
                                        batch[vector_raw[i][j]] = [dictionary1[i]]
			        else:
				        batch[vector_raw[i][j]].append(dictionary1[i])

	print 'Total epochs:', len(vocab11)
	print '________________________'
	
        counter = 0
        unique = 0
        vector_set = []
        
	for z in vocab11:
		if counter%100 == 0:
			print 'epoch number:', counter, unique
                for i in range(len(batch[z])):
                    batch[z][i] /= (numpy.linalg.norm(batch[z][i]) + 1e-5)
                
                batcharr = numpy.array(batch[z])
                neigh = NearestNeighbors(radius = 0.5)
                neigh.fit(batcharr) 
                A = neigh.radius_neighbors_graph(batcharr, mode='distance')

                model = DBSCAN(metric = 'precomputed', min_samples = 40, n_jobs = -1).fit(A)
		labels[z] = model.labels_
                
                tester = {}
                for x in labels[z]:
                    if x not in tester:
                        tester[x] = 1

                unique += len(tester)
		counter += 1
                
                del model, batch[z], tester, batcharr, neigh
                gc.collect()

	for i in range(len(vector_raw)):
		temp = []
		for j in range(len(vector_raw[i])):
                        if vector_raw[i][j] in count:
                                count[vector_raw[i][j]] += 1
			        temp.append(vector_raw[i][j] + '_' + str(labels[vector_raw[i][j]][count[vector_raw[i][j]]-1]))
                        else:
                                temp.append(vector_raw[i][j])
		vector_set.append(temp)
        
        print 'Sense vectors'

        wvec2 = Word2Vec(vector_set,  min_count = 10, size = vec_length)
        tfidf_vec2 = TfidfVectorizer(stop_words = 'english', min_df = 100,  max_features = maxfeatures)
        vector_iter2 = []
        for x in vector_set:
            vector_iter2.append(str(x))
        tfidf2 = tfidf_vec2.fit_transform(vector_iter2)
        vocab12 = wvec2.wv.vocab
        vocab22 = tfidf_vec2.get_feature_names()

        dictionary2 = numpy.zeros((len(vocab22), vec_length))
        for x in range(len(vocab22)):
                y = vocab22[x]
                if y in vocab12:
                        dictionary2[x] = wvec2[y]
        dictionary2 = tfidf2.dot(dictionary2)
        return dictionary1, dictionary2

def log_regression(input_, input__, output_):
	lr =  LogisticRegression()
	return lr.fit(input_, output_).score(input__, output_)


def accuracy(words_n, words_o):
	y_out = numpy.concatenate([numpy.zeros(12500),numpy.ones(12500)])
	print('Classifiers:')
	lr1 = log_regression(words_o[:25000], words_o[25000:], y_out)
	lr2 = log_regression(words_n[:25000], words_n[25000:], y_out)
	print 'LR_Word2Vec = ', lr1
	print 'LR_Sense2Vec = ', lr2

def removestops(data):
        data_token = [word_tokenize(x) for x in data]
        output = []
        for i in range(len(data_token)):
            x = data_token[i]
            var = []
            for j in range(len(x)):
                if x[j] not in stop_words:
                    var.append(x[j])
            output.append(var)
        return output

def main():
        print("Execution Begins:")
        data1 = preprocess_(files1, path1)
        data2 = preprocess_(files2, path2)
	data3 = preprocess_(files1, path1)
        data4 = preprocess_(files2, path2)	
        data1.extend(data2)
	data1.extend(data3)
	data1.extend(data4)

        data5 = [x[0] for x in data1]
        data = removestops(data5) 
	del data1, data2, data3, data4, data5
        words_old, words_new = create_vectors(data)
	accuracy(words_new, words_old)


if __name__ == '__main__' :
    main()

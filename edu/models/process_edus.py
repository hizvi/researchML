import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

class EDUSample:
    
    def __init__(self):
        self.nzp = (0, 0, 0) 
    
    
    def read_labeled(self, path, shuffle=True, random_state=42, z=0):
        """
        read Labeled EDUs file and prepare data for testing models
        """
        edus, targets = [], []

        label = {'p': 1, 'n': 0, 'z': None}
        
        if z == 1:
            label['n'] = -1
            label['z'] = 0
        
        # to analyze the distribution of categories
        lindex = ['n', 'z', 'p']
        nzp = [0, 0, 0]
        
        with open(path) as infile: 
            for line in infile: 
                #print(line)
                nzp[lindex.index(line[-2])] += 1
                
                # ignore neutral edus 
                if z == 0 and line[-2] == 'z':
                    continue
                
                targets.append(label[line[-2]]) 
                edus.append(line[:-2])
         
        print('=====> DATA LOADED')
        
        self.nzp = tuple(nzp)
        targets = np.array(targets)
        
        if shuffle:
            np.random.seed(random_state)
            indices = np.random.permutation(len(targets))       
            edus = [edus[i] for i in indices]
            targets = targets[indices]
        
        return edus, targets
    
    
    
    
    def split_vectorize(self, path, ngram=(1,1)):
        '''
        split data and return vectorized.
        '''
        X_train_corpus, X_test_corpus, y_train, y_test = self.split(path)
        return self.vectorize(X_train_corpus, X_test_corpus, y_train, y_test, ngram)
        
    
    def split(self, path):
        '''
        read and split data given file path
        '''
        # read form file
        edus, targets = self.read_labeled(path) 
        
        # split data
        X_train_corpus, X_test_corpus, y_train, y_test = train_test_split(edus, 
                                                                          targets, 
                                                                          test_size=1./3, 
                                                                          random_state=42)
        
        return X_train_corpus, y_train, X_test_corpus, y_test
    
    def split_data(self, edus, targets):
        '''
        split data given edus and targets
        '''
        # split data
        X_train, X_test, y_train, y_test = train_test_split(edus, 
                                                            targets, 
                                                            test_size=1./3, 
                                                            random_state=42)
        
        return X_train, y_train, X_test, y_test
    
    
    def vectorize(self, X_train_corpus, y_train, X_test_corpus, y_test, ngram=(1, 1)):
        """
        vectorize using CountVectorizer
        """
        # vectorize it
        token = r"(?u)\b[\w\'/]+\b"
        sw = stopwords.words('english').append('br')
         
        vectorizer = CountVectorizer(token_pattern=token, 
                                     min_df=5,
                                     ngram_range=ngram,
                                     stop_words=sw)
        
        X_train_vector = vectorizer.fit_transform(X_train_corpus)
        X_test_vector = vectorizer.transform(X_test_corpus)
        
        print('=====> DATA VECTORIZED')
        print('\t ngram range ', ngram)
        
        print("""                    X_train_vector shape: {}
                    y_train shape: {}
                    X_test_vector shape: {}
                    y_test shape: {}        
        """.format(X_train_vector.shape, y_train.shape, X_test_vector.shape, y_test.shape))
        
        return X_train_vector, y_train, X_test_vector, y_test
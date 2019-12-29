import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import csv
import codecs
import numpy as np

def LoadFile(path,normalisation='stemming'):
    data=pd.read_csv(path, sep=',')
    stop_words = set(stopwords.words("english"))
    text = data['text']
    ps = PorterStemmer()
    token = TweetTokenizer()
    lem = WordNetLemmatizer()
    column = []
    for i in range(len(text)):
        column.append(text[i])
        temp = []
        sentence = token.tokenize(column[i])
        for word in sentence:
            if word not in stop_words:
                if normalisation == 'stemming':
                    temp.append(ps.stem(word))
                else:
                    temp.append(lem.lemmatize(word))
        temp = " ".join(temp)
        column[i] = temp
    data['text']=column
    print(data.head())
    return data

def printGraph(x,y):
    plt.plot(x,y)
    plt.show()

def WriteSubmissionFile(path,id,data):
    df = pd.DataFrame()
    df['id'] = id
    df['target'] = data
    df.to_csv(path,index=False)

def GetLenTab(X):
    res = []
    for x in X:
        res.append(len(x))
    return np.array(res)
path = 'E:\\Kaggle\\TweetDisaster\\train.csv'
data_set = LoadFile(path,normalisation='lemmatisation')
X_train,X_valid,y_train,y_valid = train_test_split(data_set['text'],data_set['target'],shuffle=True,test_size=0.25)
token = TweetTokenizer()
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 5), tokenizer=token.tokenize,analyzer='word',max_df=0.05)
tf = TfidfTransformer()
train_text_counts = cv.fit_transform(X_train)
#p = GetLenTab(X_train)
#train_text_counts = train_text_counts.toarray()
#train_text_counts= np.insert(train_text_counts,train_text_counts.shape[1],p,axis=1)
train_text_counts = tf.fit_transform(train_text_counts)
valid_text_counts = cv.transform(X_valid)
#p = GetLenTab(X_valid)
#valid_text_counts = valid_text_counts.toarray()
#valid_text_counts = np.insert(valid_text_counts,valid_text_counts.shape[1],p,axis=1)
valid_text_counts = tf.transform(valid_text_counts)

x_alpha = np.arange(0,1,0.01)
y_alpha = []
for a in x_alpha:
    clf = MultinomialNB(alpha=a).fit(train_text_counts, y_train)
    predicted= clf.predict(valid_text_counts)
    score = metrics.accuracy_score(y_valid, predicted)
    print("MultinomialNB Accuracy:", score)
    y_alpha.append(score)
printGraph(x_alpha,y_alpha)
path = 'E:\\Kaggle\\TweetDisaster\\test.csv'
test_set = LoadFile(path)
test_text_counts = cv.transform(test_set['text'])
#test_text_counts = tf.transform(test_text_counts)
clf = MultinomialNB(alpha=0.8).fit(train_text_counts, y_train)
predicted= clf.predict(test_text_counts)
path = 'E:\\Kaggle\\TweetDisaster\\submission1.csv'
WriteSubmissionFile(path,test_set['id'],predicted)

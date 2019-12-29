import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
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
    lem = WordNetLemmatizer()
    column = []
    for i in range(len(text)):
        column.append(sent_tokenize(text[i]))
        temp = []
        for j in range(len(column[i])):
            sentence = word_tokenize(column[i][j])
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

def WriteSubmissionFile(path,id,data):
    df = pd.DataFrame()
    df['id'] = id
    df['target'] = data
    df.to_csv(path,index=False)

path = 'E:\\Kaggle\\TweetDisaster\\train.csv'
train_set = LoadFile(path)
path = 'E:\\Kaggle\\TweetDisaster\\test.csv'
test_set = LoadFile(path)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
train_text_counts = cv.fit_transform(train_set['text'])
test_text_counts = cv.transform(test_set['text'])
clf = MultinomialNB(alpha=0.2).fit(train_text_counts, train_set['target'])
predicted= clf.predict(test_text_counts)
path = 'E:\\Kaggle\\TweetDisaster\\submission1.csv'
WriteSubmissionFile(path,test_set['id'],predicted)
#print("MultinomialNB Accuracy:",metrics.accuracy_score(train_set['target'], predicted))
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all the required packages and Libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# In[2]:


# Reading a  text file, data=pandas.read_csv(‘filename.txt’, sep=’ ‘, header=None, names=[“Column1”, “Column2”])
df =  pd.read_csv("SMSSpamCollection",sep ="\t", names = ["Label", "message"] )


# In[3]:


# see the first 5 rows in dataframe
df.head()


# In[4]:


# Import Stopwords
stop_words = stopwords.words("english")


# In[5]:


# Cleaning the messages
stemmer = PorterStemmer()
corpus = []
def Cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', text) # Removing all the punctuations and numbers
    text= text.lower() # Converting text to lower case
    text = text.split() # Splitting the sentences to words to remove stop words
    text = [w for w in text if w not in stop_words] # Removing all the stopwords
    text = [stemmer.stem(w) for w in text] # Stemming words
    text = ' '.join(text) # Joining steemed words 
    corpus.append(text) # Storing clean text sentences in a list
    return text

df["clean_text"] = df["message"].apply(Cleaning)


# In[7]:


# see the first 5 rows in dataframe
df.head()


# In[26]:


# VBisulaising corpus
print(corpus)


# In[9]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
X = cv.fit_transform(corpus).toarray()


# In[10]:


# Visualising X
print(X)


# In[17]:


# Encoding Label column
df = df.replace({"Label":{"ham":"0", "spam":"1"}})


# In[20]:


# tarin_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, df["Label"], random_state = 42, test_size = 0.2)


# ## Logistic Regression

# In[24]:


# Training Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
model.score(x_test, y_test)


# ## SVM

# In[27]:


from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
y_prediction = svm_model.predict(x_test)
svm_model.score(x_test, y_test)


# ## Random_Forest_Classifier

# In[29]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
clf.score(x_test, y_test)


#!/usr/bin/env python
# coding: utf-8

# ### Required Libraries and Packages 

# In[2]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')
get_ipython().system('pip install nltk')


# In[3]:


import re
import spacy
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec


# In[4]:


data = pd.read_csv('/Users/fathhi/Desktop/MSc/Semester2/7120CEM_NLP/CW/CW1_DataSet.csv')


# In[5]:


print(data.head())


# ### Preprocessing the text

# In[6]:


nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    
    text = text.lower()
    
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = word_tokenize(text)
   
    tokens = [word for word in tokens if word not in stop_words]
    
    tokens = [nlp(word)[0].lemma_ for word in tokens]
    
    return ' '.join(tokens)


data['processed_text'] = data['text'].apply(preprocess_text)


print(data.head())


# In[7]:


data.drop('text', axis=1, inplace=True)
print(data.head())


# ### Defining and Labeling the target variable

# In[8]:


def label_emotion(text):
    text_lower = text.lower()  
    
    
    if any(keyword in text_lower for keyword in ['angry', 'attack', 'assault']):
        return 'angry'
    elif any(keyword in text_lower for keyword in ['happy', 'celebrate', 'joy']):
        return 'happy'
    elif any(keyword in text_lower for keyword in ['sad', 'tragedy', 'grief']):
        return 'sad'
    else:
        return 'neutral'


data['emotion'] = data['processed_text'].apply(label_emotion)


print(data.head())


# In[9]:


emotion_counts = data['emotion'].value_counts()


emotion_ratios = emotion_counts / len(data)


plt.figure(figsize=(8, 6))
plt.pie(emotion_ratios, labels=emotion_ratios.index, autopct='%1.1f%%', startangle=140)
plt.title('Emotion Distribution')
plt.axis('equal')  
plt.show()


emotion_counts = data['emotion'].value_counts()

emotion_ratios = emotion_counts / len(data)

print("Emotion Distribution:")
print(emotion_ratios)


# In[10]:


emotion_counts = data['emotion'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(emotion_counts.index, emotion_counts.values)
plt.title('Distribution of Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# ### Pre-processing for balanced data

# In[11]:


nltk.download('wordnet')
nltk.download('omw-1.4')  


def get_synonyms(words):
    synonyms = set()
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
    return list(synonyms)


angry_keywords = ['angry', 'attack', 'assault']
happy_keywords = ['happy', 'celebrate', 'joy']
sad_keywords = ['sad', 'tragedy', 'grief']


extended_happy_keywords = get_synonyms(happy_keywords)
extended_sad_keywords = get_synonyms(sad_keywords)


happy_keywords.extend(extended_happy_keywords)
sad_keywords.extend(extended_sad_keywords)


def label_emotion(text):
    text_lower = text.lower()  
    
   
    if any(keyword in text_lower for keyword in angry_keywords):
        return 'angry'
    elif any(keyword in text_lower for keyword in happy_keywords):
        return 'happy'
    elif any(keyword in text_lower for keyword in sad_keywords):
        return 'sad'
    else:
        return 'neutral'


data['emotion'] = data['processed_text'].apply(label_emotion)


emotion_counts = data['emotion'].value_counts()


emotion_ratios = emotion_counts / len(data)


plt.figure(figsize=(8, 6))
plt.pie(emotion_ratios, labels=emotion_ratios.index, autopct='%1.1f%%', startangle=140)
plt.title('Emotion Distribution')
plt.axis('equal')  
plt.show()


emotion_counts = data['emotion'].value_counts()


emotion_ratios = emotion_counts / len(data)

print("Emotion Distribution:")
print(emotion_ratios)


# ### Word2Vec Feature Extraction

# In[13]:


import numpy as np


corpus = [text.split() for text in data['processed_text']]


word2vec_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4)


def get_avg_word2vec(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


word2vec_features = data['processed_text'].apply(lambda x: get_avg_word2vec(x, word2vec_model))


word2vec_df = pd.DataFrame(word2vec_features.tolist())


df_with_word2vec = pd.concat([data[['article_id', 'processed_text', 'emotion']], word2vec_df], axis=1)


print(df_with_word2vec.head())


# In[14]:


X = df_with_word2vec.drop(['article_id', 'processed_text', 'emotion'], axis=1)  
y = df_with_word2vec['emotion']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Decision Tree with Hyperparameter Tuning

# In[15]:


param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


dt_clf = DecisionTreeClassifier(random_state=42)


grid_search_dt = GridSearchCV(dt_clf, param_grid_dt, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)


grid_search_dt.fit(X_train, y_train)


best_params_dt = grid_search_dt.best_params_
best_score_dt = grid_search_dt.best_score_

print("Best Parameters (Decision Tree):", best_params_dt)
print("Best Cross-validation Accuracy (Decision Tree):", best_score_dt)


dt_best = DecisionTreeClassifier(**best_params_dt, random_state=42)
dt_best.fit(X_train, y_train)


y_pred_dt_best = dt_best.predict(X_test)


accuracy_dt_best = accuracy_score(y_test, y_pred_dt_best)
print("Decision Tree Best Accuracy:", accuracy_dt_best)
print("Decision Tree Best Classification Report:")
print(classification_report(y_test, y_pred_dt_best))


# ### Random Forest 

# In[16]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))


# ### Hyperparameter tuning with Random Forest

# In[17]:


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


rf_model_resampled = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_resampled.fit(X_resampled, y_resampled)


y_pred_rf_resampled = rf_model_resampled.predict(X_test)


accuracy_rf_resampled = accuracy_score(y_test, y_pred_rf_resampled)
print("Random Forest Accuracy after Resampling:", accuracy_rf_resampled)
print("Random Forest Classification Report after Resampling:")
print(classification_report(y_test, y_pred_rf_resampled, target_names=label_encoder.classes_))


# ### XGboost Algorithm with Hyperparameter tuning

# In[19]:


X = df_with_word2vec.drop(['article_id', 'processed_text', 'emotion'], axis=1)
y = df_with_word2vec['emotion']


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)


y_pred_xgb = xgb_model.predict(X_test)


xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)


grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)


grid_search_xgb.fit(X_train, y_train)


best_params_xgb = grid_search_xgb.best_params_
best_xgb_model = grid_search_xgb.best_estimator_


y_pred_xgb_best = best_xgb_model.predict(X_test)


best_xgb_accuracy = accuracy_score(y_test, y_pred_xgb_best)
print("Best XGBoost Accuracy:", best_xgb_accuracy)
print("Best XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb_best, target_names=label_encoder.classes_))


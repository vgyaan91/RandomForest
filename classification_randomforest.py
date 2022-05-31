import os
import pandas as pd
from bs4 import BeautifulSoup
disease_data=[]
for filename in os.listdir("C:/Users/Gyan/Downloads/disease_data/short_data/disease"):
   with open(os.path.join("C:/Users/Gyan/Downloads/disease_data/short_data/disease", filename), 'r', encoding="utf8") as f:
       text = f.read()
       print(text)
       disease_data.append(text)
       

disease_data_new= disease_data.copy()      

title_text=[]
for i in disease_data_new:
    text=[]
    title= BeautifulSoup(i, 'html.parser')
    text.append(title.title.get_text())
    title_text.append(text)
print(title_text)

title_lst= []
for i in title_text:
    new= " ".join(i)
    title_lst.append(new.split("-")[0])

print(title_lst[0:6])

p_text=[]
for i in disease_data_new:
    text=[]
    p= BeautifulSoup(i, 'html.parser')
    for ptag in p.find_all("p"):
        text.append(ptag.get_text())
    p_text.append(text)
print(p_text)

p_lst= []
for i in p_text:
    p_lst.append(" ".join(i))
    
print(p_lst[0])

disease_df= pd.DataFrame({'Article': p_lst,'Title':title_lst})
disease_df['Category']= "Disease Article"
disease_df.head(10)



noise_data=[]
for filename in os.listdir("C:/Users/Gyan/Downloads/disease_data/short_data/others"):
   with open(os.path.join("C:/Users/Gyan/Downloads/disease_data/short_data/others", filename), 'r', encoding="utf8") as f:
       text1 = f.read()
       print(text1)
       noise_data.append(text1)
       

noise_data_new= noise_data.copy()      

other_title_text=[]
for i in noise_data_new:
    text=[]
    title= BeautifulSoup(i, 'html.parser')
    text.append(title.title.get_text())
    other_title_text.append(text)
print(other_title_text)

other_title_lst= []
for i in other_title_text:
    new= " ".join(i)
    other_title_lst.append(new.split("-")[0])
    
print(other_title_lst[0])

other_p_text=[]
for i in noise_data_new:
    text=[]
    p= BeautifulSoup(i, 'html.parser')
    for ptag in p.find_all("p"):
        text.append(ptag.get_text())
    other_p_text.append(text)
print(other_p_text)

other_p_lst= []
for i in other_p_text:
    other_p_lst.append(" ".join(i))
    
print(other_p_lst[0])

other_df= pd.DataFrame({'Article': other_p_lst,'Title':other_title_lst})
other_df['Category']= "Other Article"
other_df.head(10)

final_data=pd.concat([disease_df, other_df])

final_data.head(10)
final_data.tail(10)


X= final_data['Article']
y= final_data['Category']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)


X_train.reset_index(inplace=True,drop=True)

import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

nltk.download('stopwords')
stop= set(stopwords.words('english'))

cleaned_train=[]
for i in range(0, X_train.shape[0]):
  text=re.sub('[^a-zA-Z]', ' ', X_train[i])
  text=text.lower()
  text=text.split()
  text= [ x for x in text if not x in stop]
  porterstem= PorterStemmer()
  stemmed_tokens= [porterstem.stem(words) for words in text]
  cleaned_train.append(' '.join(stemmed_tokens))
  
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf = TfidfVectorizer()
X_train= tfidf.fit_transform(cleaned_train)

from sklearn.ensemble import RandomForestClassifier
# rfmodel= RandomForestClassifier(n_estimators=40, oob_score=True, n_jobs=-1,
#                                 random_state=101, max_features="auto", min_samples_leaf=20)



# rfmodel.fit(X_train, y_train)
# # y_pred= rfmodel.predict(X_test)

# def inference(data,predictor):
#     testbag=[]
#     for i in range(0,data.shape[0]):
#         text=re.sub('[^a-zA-Z]',' ',data[i])
#         text=text.lower()
#         text=text.split()
#         text= [ w for w in text if not w in stop]
#         ps =PorterStemmer()
#         stemmed_tokens= [ps.stem(words) for words in text]
#         testbag.append(' '.join(stemmed_tokens))
#     X_test= tfidf.transform(testbag)
#     pred= predictor.predict(X_test)
#     return pred

# pred= inference(X_test, rfmodel)
# from sklearn.metrics import confusion_matrix, classification_report

# print(classification_report(y_test,pred))

# X_test[0]

# pred[0]


# Using Grid Search to find the best Parameters
from sklearn.model_selection import GridSearchCV
rfmodel= RandomForestClassifier(random_state=12)
grid_param= {
    "n_estimators": [40, 80, 100, 120, 130],
    "criterion": ["gini", "entropy"],
    "max_depth": range(2, 10, 1),
    "min_samples_leaf": range(1, 10, 1),
    "min_samples_split": range(2, 10, 1),
    "max_features": ["auto", "log2"]}

grid_search= GridSearchCV(estimator= rfmodel, param_grid= grid_param, cv=5, n_jobs=-1, verbose=3)

grid_search.fit(X_train, y_train)

grid_search.best_params_

final_rfmodel= RandomForestClassifier(criterion= 'gini', max_depth= 9, max_features= 'auto',
                                      min_samples_leaf= 2, min_samples_split= 8, n_estimators= 40,
                                      random_state=12)

model_fit= final_rfmodel.fit(X_train, y_train)

X_test.reset_index(inplace=True,drop=True)

cleaned_test=[]
for i in range(0, X_test.shape[0]):
  text=re.sub('[^a-zA-Z]', ' ', X_test[i])
  text=text.lower()
  text=text.split()
  text= [ x for x in text if not x in stop]
  porterstem =PorterStemmer()
  stemmed_tokens= [porterstem.stem(words) for words in text]
  cleaned_test.append(' '.join(stemmed_tokens))
  
X_test= tfidf.transform(cleaned_test)

final_rfmodel.score(X_test, y_test)


pred= model_fit.predict(X_test)

print(cleaned_test[0])
pred[0]

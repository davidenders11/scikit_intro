import numpy as np
import pandas as pd # dataset manipulation
import matplotlib.pyplot as plt # data visualization
from imblearn.under_sampling import  RandomUnderSampler # handle imbalanced data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# import data, then create an imbalanced subset for training purposes
df_review = pd.read_csv('IMDB_Dataset.csv')
df_positive = df_review[df_review['sentiment']=='positive'][:9000]
df_negative = df_review[df_review['sentiment']=='negative'][:1000]
df_review_imb = pd.concat([df_positive, df_negative]) 

# df_review_imb['sentiment'].value_counts().plot(kind='bar')
# plt.show() # use this to visualize

# use imblearn to balance data
rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review_imb[['review']],
                                                           df_review_imb['sentiment'])
# dataFrame is now balanced

# use 2/3 of data for training, 1/3 for testing/validating
train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42) 
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# we are using term frequency/document frequency to identify unique words for sentiment
tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
test_x_vector = tfidf.transform(test_x)

# using SVM model for supervised classification
svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)
print(svc.score(test_x_vector, test_y))
print(svc.predict(tfidf.transform(["I almost peed myself watching this move. Solid use of my time."])))
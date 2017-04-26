# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:54:29 2017

@author: Andrea

-------------------------------------------------------------------------------
This is the first code I upload into the kaggle platform.
I am starting with the Titanic competition. 
References for this study are the many codes already submitted for the competition
and the text book "Python Machine Learning" from Sebastain Raschka.
"""

# Import libraries for data analysis and machine learning
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# Preprocessing data
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
# Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# Metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, confusion_matrix
# Tuning hyperparameters
from sklearn.grid_search import GridSearchCV
# Diagnostic curves
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

# Step 1: uploading the dataset into a Pandas dataframe
df_train = pd.read_csv("./Input/train.csv")
df_test = pd.read_csv("./Input/test.csv")
# Identify the structure of the dataset showing its first 5 rows
df_train.head()
df_train.columns 
# Class label is Survived: 1 = survived; 0 = Not-survived 
# From the inspection of the data we can see that the following features are 
# numeric:
#   - Pclass, Age, SibSp, Parch, Fare
# and the following features are categorical:
#   - Name, Sex, Ticket, Cabin, Embarked
#
# PREPROCESSING DATA:
# Check the number of samples in the training dataset
df_train.shape[0]  
# Let's now identify the missing data
df_train.isnull().sum()
# Embarked has 2 (out of 891) missing data
# --> for the Embarked feature the missing data will be replaced by the most 
#       frequent value in the feature:
embarked_labels = df_train['Embarked'].unique()
total_emb_label = {}
for label in embarked_labels:
    total_emb_label[label] =  \
        df_train.loc[df_train['Embarked']==label, 'Embarked'].count()
    print('Port %s: %d passengers embarked' % (label, total_emb_label[label]))
# Most of the passengers boarded at 'S' (Southampton) port. 'S' will be assigned 
# to the missing data:
df_train['Embarked'] = df_train['Embarked'].fillna('S') 
df_train['Embarked'].value_counts()
#
# Age has 177 (out of 891) missing data
print('Data missing for Age feature: %.3f' % 
      (df_train['Age'].isnull().sum()/df_train.shape[0]))
# 19.9% of the Age data is missing.
# Let's identify the maximum and minimum age:
df_train['Age'].max()
df_train['Age'].min()
temp_age = df_train['Age'].dropna()
# Check the distribution of the age for the passengers we have information for
plt.hist(x=temp_age, bins=16, label='Age distribution')
plt.xlabel('Age [-]')
plt.ylabel('Frequency [-]')
plt.title('Distribution of age of passangers of Titanic')
plt.show()
temp_age.describe()
# The histogram shows that most of the passengers were between 20 and 35 years old
# The missing data for age will be replaced by the median of the Age feature
imr = Imputer(missing_values="NaN", strategy="median", axis=0)
imr.fit(df_train['Age'].values.reshape(-1,1))
df_train['Age'] = imr.transform(df_train['Age'].values.reshape(-1,1))
# Cabin has 687 (out of 891) missing data --> Too much data missign ==> 
# It is not possible to make assumptions regarding the missing data
""" Machine Learning"""
# Divide the dataset into training data X and training class label y:
X_train = df_train.iloc[:, 2:] # Passenger ID is not taken into account for the analysis
y_train = df_train.iloc[:,1]
# Scale the data

# Let's start with a simple linear classifier: Perceptron
#ppn = Perceptron(penalty=None, n_iter=40, eta0 = 0.1, random_state=0)
#ppn.fit(X_train, y_train)

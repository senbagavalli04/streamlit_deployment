import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.read_csv("C:/Users/D E L L/Downloads/Ml assignment II/tic-tac-toe.csv")

# Encoding categorical features using OneHotEncoder
X = pd.get_dummies(data.drop('class', axis=1))
y = data['class']

X=X.astype(int)
y=y.astype(int)

# Encoding categorical labels using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Save the model again with a compatible protocol
with open("pick1_new.sav", "wb") as file:
    pickle.dump(random_forest, file)

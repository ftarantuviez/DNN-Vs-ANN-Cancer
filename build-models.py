import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=.3, random_state=42)

# ANN
ANN = Sequential()
ANN.add(Dense(15, input_dim=30, activation="relu"))
ANN.add(Dense(1, activation="sigmoid"))
ANN.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
ANN_history = ANN.fit(X_train, y_train, epochs=20, batch_size=50)

ann_history_file = open("ANN_history.json", "w")
json.dump(ANN_history.history, ann_history_file)
ann_history_file.close()

DNN.save("ANN_model")

# DNN
DNN = Sequential()
DNN.add(Dense(15, input_dim=30, activation="relu"))
DNN.add(Dense(15, activation="relu"))
DNN.add(Dense(15, activation="relu"))
DNN.add(Dense(1, activation="sigmoid"))
DNN.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

DNN_history = DNN.fit(X_train, y_train, epochs=20, batch_size=50)

dnn_history_file = open("DNN_history.json", "w")
json.dump(DNN_history.history, dnn_history_file)
dnn_history_file.close()

DNN.save("DNN_model")





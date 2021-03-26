import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import streamlit as st
import json
import functions_utils as fu
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='ANN vs DNN', page_icon="./f.png")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.title('Artificial Neural Network vs Deep Neural Network')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
st.write('---')
st.write("""
The idea of this app is compare the results of prediction between a Deep Neural Network against a Simple Neural Network.
The project consist in predict if the patient has cancer or if does not have cancer.The data that we are gonna use is provided by Scikit-learn in their package called: "load_breast_cancer". \n
**WARNING**: The idea of this post is not explain what an Artificial Neural Network (ANN) is, but compare it results with a Deep Neural Network (DNN). However, in case you are interested on know more about these magnificent mathematical advances, I'll give you some resources to visit at the bottom of this page.
""")

image = Image.open('ann.png')
st.image(image, caption='Artificial Neural Network Representation')


# Loading Data

st.write(""" 
## Loading Data

First we load the data. \n
```
df = sklearn.datasets.load_breast_cancer()
``` \n
Loading this dataset provides us an dictionary which two keys: "data" and "target":
```
 X = df.data \n
 Y = df.target
```
The variable `X` is the data that we are gonna use as features to fit and train the model. `Y` is a binary feature which depicts if the patient has cancer or not (1 if yes, 0 if no).

And finally, we just separate into training data and test data:

```
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=42)
```
\n
""")

df = load_breast_cancer()
X = df.data
Y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=42)

st.write(""" 
After 
```
X.head()
```
we get:
""")


st.dataframe(pd.DataFrame(X).head())
# / Loading Data

# Building ANN

st.write(""" 
## Artificial Neural Network \n

First we are gonna compute a Simple Neural Network with only one hidden layer. For all the project (both for the ANN and for the DNN) we use Keras as library with Tensorflow running in the backend.

### Building and Fitting the ANN

This ANN will be a Sequential Neural Network with a single hidden layer containing 15 neurons (units). As activation function in the hidden layer we use ReLU, and for the output layer we use the classic sigmoid function.

```
ANN = Sequential()
ANN.add(Dense(15, input_dim=30, activation="relu"))
ANN.add(Dense(1, activation="sigmoid"))
ANN.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

```
""")


st.line_chart(pd.read_json("ANN_history.json"))

st.write(""" 
Then we use the test data to extract another kind of metrics from our previously trained model:
```
predictions = ANN.predict(X_test)
ann_confusion_matrix = metrics.confusion_matrix(y_pred=predictions, y_true=y_test)
```
"""
)

# confusion matrix of ANN (extracted from notebook)
fu.plot_confusion_matrix([[59, 4], [8, 1e+02]], "ANN")

# / Building ANN

# Building DNN

st.write("""
## Deep Neural Network

The difference between a Simple Artificial Neural Network and a Deep Neural Network is the quantity of hidden layer of them. There is no a standard of how many hidden layers should the network have to be considered a DNN, but usually when one Neural Network has more than 1 hidden layer is cataloged as a deep network.

In this case we are gonna use three hidden layers with 15 units each one. For all these layers we set the activation function as the ReLU function, and for the output layer, the sigmoid one. Also, as optimizer we are gonna use *rmsprop* and as metrics the accuracy.

""")
st.code("""
DNN = Sequential()
DNN.add(Dense(15, input_dim=30, activation="relu"))
DNN.add(Dense(15, activation="relu"))
DNN.add(Dense(15, activation="relu"))
DNN.add(Dense(1, activation="sigmoid"))
DNN.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
DNN.fit(X_train, y_train)
""")

st.line_chart(pd.read_json("DNN_history.json"))

# confusion matrix of DNN (extracted from notebook)
fu.plot_confusion_matrix([[57, 6], [1, 1.1e+02]], "DNN")

# / Building DNN

# Conclusion
st.write("""
## Conclusion

In the below dataframe, we see the accuracy of the two models: the ANN accuracy and the DNN accuracy. Is easy to note that the last model have better metrics and performance than the first one. However, not always happen this. We cannot say that having more complex neural networks we are gonna have better results. Indeed, we can fall in the overfittin problem, or in a processing problem (the computer have to process a lot more in a complex net than in a simple one).

#### Accuracy
"""
)

# Accuracy of ANN and DNN (extracted from notebook)
metrics_dataframe = pd.DataFrame([0.9298, 0.9591]).T.rename({0: "ANN", 1: "DNN"}, axis=1)
st.dataframe(metrics_dataframe)

st.write(""" 
### Resources of Artificial Neural Networks and Deep Neural Networks:
* [But, what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk)
* [Gradient Descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
* [Artificial Neural Network - Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network)
* [Artificial Neural Network - an overview | ScienceDirect Topics](https://www.sciencedirect.com/topics/engineering/artificial-neural-network)

And on Twitter, [svpino](https://twitter.com/svpino) can guide you better than anyone to understand how to start with Machine Learning and AI.
""")

# / Conclusion

# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/DNN-vs-ANN)
""")
# / This app repository


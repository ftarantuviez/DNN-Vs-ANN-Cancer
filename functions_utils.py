import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_loss_accuracy(history, col=None):
  plt.rcParams.update({'font.size': 8})
  plt.figure(figsize=(4,4))
  plt.plot(history["loss"], label="Loss")
  plt.plot(history["accuracy"], label="Accuracy")
  plt.xticks(range(0,21, 5))
  plt.xlabel("Epoch")
  plt.title("Loss vs Accuracy")
  plt.legend()
  return st.pyplot()

def plot_confusion_matrix(matrix, model_name, col=None):
  sns.heatmap(matrix, annot=True)
  plt.title("Confusion matrix for {}".format(model_name))

  if col:
    return col.pyplot()
  return st.pyplot()
# Exoplanet Detection with a Convolutional Neural Network

This project demonstrates the application of a 1D Convolutional Neural Network (CNN) for detecting exoplanets using data from the Kepler Space Telescope. The model analyzes light curve data to identify the characteristic dips in brightness that indicate the presence of an orbiting planet.

## Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training Process](#training-process)
* [Evaluation Metrics](#evaluation-metrics)
* [Dependencies](#dependencies)


## Introduction

Detecting exoplanets is a challenging task, traditionally relying on computationally intensive methods. This project leverages the power of deep learning to automate and improve the accuracy of exoplanet detection. The CNN model learns to recognize patterns in light curve data that are indicative of planetary transits. Using data collected by the Kepler Space Telescope, we can create a prediction which is 96% accurate on whether an exoplanet is circulating a star 🌟


## Dataset

The project uses the Kepler cumulative table (`cumulative.csv`) available [here](https://docs.google.com/spreadsheets/d/1FXjO1HUBDWklhtGZ0h60l1s4NJtU0bpkgdCQ_DJQsOo/edit?usp=sharing). This dataset contains time-series light curve data along with various stellar and planetary parameters for thousands of observed stars.  Preprocessing steps include:

* Removing irrelevant columns (e.g., star names, Kepler IDs).
* Handling missing values (currently using removal via `dropna()`, but other strategies can be explored).
* Encoding the target variable `koi_pdisposition` (using Label Encoding).
* Converting data to NumPy arrays for compatibility with PyTorch.



## Model Architecture

The model is a 1D CNN consisting of:

* Two convolutional layers (with ReLU activation) followed by max-pooling layers.
* Two fully connected layers (with ReLU activation on the first).
* An output layer with log-softmax activation for binary classification (planet or no planet).


(Include a diagram or detailed description of the layers, filters, kernel size, etc.  If you used a well-known architecture like ResNet or Inception, mention it.)



## Training Process


The model is trained using 25-fold cross-validation to ensure robust performance and minimize overfitting. The Adam optimizer is used with a learning rate of 0.002. Cross-entropy loss is used as the loss function. Each fold is trained for 20 epochs using the full dataset as a batch.  (Mention early stopping if used).


## Evaluation Metrics


The model's performance is evaluated using the following metrics:

* Accuracy: Overall correctness of classifications.
* Precision: Proportion of correctly identified planets out of all predicted planets.
* Recall: Proportion of correctly identified planets out of all actual planets.
* F1-Score: Harmonic mean of precision and recall.
* Confusion Matrix: Visualization of true positives, true negatives, false positives, and false negatives.
* (AUC, ROC Curve if calculated)


## Dependencies

* Python 3.x
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn


# Exoplanet Detection with a Convolutional Neural Network

This project demonstrates the application of a 1D Convolutional Neural Network (CNN) for detecting exoplanets using data from the Kepler Space Telescope. The model analyzes light curve data to identify the characteristic dips in brightness that indicate the presence of an orbiting planet.

## Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training Process](#training-process)
* [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [Conclusion and Future Work](#conclusion-and-future-work)
* [Getting Started](#getting-started)
* [Dependencies](#dependencies)


## Introduction

Detecting exoplanets is a challenging task, traditionally relying on computationally intensive methods. This project leverages the power of deep learning to automate and improve the accuracy of exoplanet detection. The CNN model learns to recognize patterns in light curve data that are indicative of planetary transits.


## Dataset

The project uses the Kepler cumulative table (`cumulative.csv`) available [here](link-to-dataset). This dataset contains time-series light curve data along with various stellar and planetary parameters for thousands of observed stars.  Preprocessing steps include:

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



## Results

(Summarize your findings. Include the average accuracy across all folds, the confusion matrix, and other metrics. Discuss any insights from the results.  Include graphs/visualizations if possible).


## Conclusion and Future Work

(Summarize the project's achievements and limitations. Discuss potential future improvements, such as using different CNN architectures, hyperparameter optimization, or alternative data preprocessing strategies.)



## Getting Started


1. Clone this repository: `git clone https://github.com/your-username/exoplanet-detection-cnn.git`
2. Install the required dependencies (see below).
3. Download the dataset (`cumulative.csv`) and place it in the project directory.
4. Run the Jupyter notebook: `jupyter notebook exoplanet_detection.ipynb`



## Dependencies

* Python 3.x
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn


(Specify version numbers for key libraries if you've pinned them for reproducibility).

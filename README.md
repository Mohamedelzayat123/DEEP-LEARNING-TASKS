# Iris Classification with Neural Networks

## Objective
The objective of this task is to classify iris flowers into three distinct species—*Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*—using a neural network built with PyTorch. The goal is to achieve an accuracy of at least 95% using a well-prepared dataset, with metrics such as accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC curve evaluated to assess performance.

## Dataset Description
The Iris dataset is a classic dataset in machine learning and statistics. It consists of 150 samples of iris flowers, each described by four features:

- **Sepal length (cm)**
- **Sepal width (cm)**
- **Petal length (cm)**
- **Petal width (cm)**

The target variable represents the flower species, which can be one of the following:

1. **Iris-setosa**
2. **Iris-versicolor**
3. **Iris-virginica**

### Dataset Breakdown:
- Features: 4 continuous variables
- Classes: 3 classes (Iris-setosa, Iris-versicolor, Iris-virginica)
- Samples: 150 total samples

## Steps to Run the Code in Jupyter

### 1.  Install the Dependencies
To get started, first clone this GitHub repository:

```bash


### 1. Install the Dependencies

pip install torch
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tqdm

### 2. Open the Jupyter Notebook

jupyter notebook

### 3. Run the Notebook
Open the notebook file (iris_classification.ipynb) and run each cell sequentially. The code is organized into sections:

Data Preparation: Load and preprocess the Iris dataset.
Data Visualization: Visualize the features of the dataset.
Neural Network Configuration: Set up a neural network using PyTorch.
Model Training: Train the neural network on the dataset.
Evaluation Metrics: Evaluate the model using accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC curve.
Plotting: Visualize the training accuracy and loss, and plot the ROC-AUC curves for each class.


# Facial Recognition with PINS Dataset

This repository contains a project to classify images of faces into different categories based on the person's identity using a multiclass classification model built with Keras. The goal is to achieve an accuracy of 85% or higher on the validation/test dataset.

## Objective
The primary objective of this project is to develop a deep learning model to classify facial images of celebrities (or individuals) from the PINS dataset into their respective categories.

## Dataset
The dataset used for this project is the [Pins Face Recognition Dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition), which contains multiple facial images of various individuals. Each class corresponds to a different person.

### Dataset Details:
- Contains images of celebrities or individuals.
- Images are grouped by person (class).

## Requirements

### Dependencies:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV (optional for face cropping)
- matplotlib
- scikit-learn

To install the dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Project Structure
```
.
├── dataset/                 # Folder for the dataset
├── models/                  # Folder for saving trained models
├── notebooks/               # Jupyter Notebooks for the project
│   ├── face_classification.ipynb  # Main notebook for the project
├── README.md                # Project description and instructions
```

## Steps

### 1. Data Loading and Preprocessing
- Load the dataset using Keras' `ImageDataGenerator`.
- Resize images to 100x100 pixels for uniformity.
- Normalize pixel values to [0, 1].
- Split the dataset into training and validation sets (80-20 split).
- (Optional) Apply data augmentation (random flips, rotations, zooms).
- (Optional) Use OpenCV DNN to crop images to faces only.

### 2. Model Creation (ANN)
- Build an artificial neural network (ANN) using the Keras Sequential API.
- Network architecture:
  - Input layer for 100x100 images flattened into 10,000 features.
  - Hidden layers with 512, 256, and 128 neurons (ReLU activation).
  - Output layer with softmax activation (number of neurons = number of classes).
- Regularization techniques like dropout or batch normalization to prevent overfitting.
- Use a suitable optimizer (e.g., Adam) and loss function (e.g., `categorical_crossentropy`).

### 3. Training
- Train the model on the training dataset.
- Use validation dataset for monitoring.
- Apply early stopping and learning rate scheduling as needed.

### 4. Evaluation
- Evaluate the model on the test dataset.
- Generate metrics like accuracy, precision, recall, and F1-score.
- Visualize training/validation accuracy and loss curves.
- Plot confusion matrix.


## Instructions for Running the Code

1. Clone the repository:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd <project_directory>
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Jupyter Notebook:

```bash
jupyter notebook notebooks/face_classification.ipynb
```

5. Follow the instructions in the notebook to execute the data preprocessing, model creation, training, and evaluation steps.

## Visualizations
- Training and validation loss and accuracy curves.
- Confusion matrix for classification results.

## Deliverables
- Jupyter Notebook (`7538_bonus.ipynb`) containing the code and explanations.
- Organized repository structure for ease of use.


### Author
[MOHAMED MOHAMED MAHMOUD]


---


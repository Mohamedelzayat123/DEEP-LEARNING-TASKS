# Heart Disease Prediction Using MLP

## Objective
The objective of this project is to build a Multi-Layer Perceptron (MLP) model using TensorFlow 2.0 to predict heart disease based on various health and demographic features. The model will be evaluated using several metrics, including accuracy, confusion matrix, precision, recall, F1-score, and ROC-AUC curve.

## Dataset Description
The dataset used in this project is derived from the `heart.csv` file. It contains health and demographic information about individuals, including features such as:
- **HeartDiseaseorAttack**: Target variable indicating the presence of heart disease (1 = Yes, 0 = No).
- **HighBP**: Indicator of high blood pressure (1 = Yes, 0 = No).
- **HighChol**: Indicator of high cholesterol (1 = Yes, 0 = No).
- **CholCheck**: Indicator of cholesterol check (1 = Yes, 0 = No).
- **BMI**: Body Mass Index.
- **Smoker**: Indicator of smoking status (1 = Yes, 0 = No).
- **Stroke**: Indicator of stroke history (1 = Yes, 0 = No).
- **Diabetes**: Indicator of diabetes status (1 = Yes, 0 = No).
- **PhysActivity**: Indicator of physical activity (1 = Yes, 0 = No).
- **Fruits**: Indicator of fruit consumption (1 = Yes, 0 = No).
- **Veggies**: Indicator of vegetable consumption (1 = Yes, 0 = No).
- **HvyAlcoholConsump**: Indicator of heavy alcohol consumption (1 = Yes, 0 = No).
- **AnyHealthcare**: Indicator of healthcare access (1 = Yes, 0 = No).
- **NoDocbcCost**: Indicator of cost-related doctor visits (1 = Yes, 0 = No).
- **GenHlth**: General health rating (1-5).
- **MentHlth**: Mental health rating (number of days).
- **PhysHlth**: Physical health rating (number of days).
- **DiffWalk**: Indicator of difficulty walking (1 = Yes, 0 = No).
- **Sex**: Gender (1 = Male, 0 = Female).
- **Age**: Age of the individual.
- **Education**: Education level.
- **Income**: Income level.

## Steps to Run the Code in Jupyter Notebook
1. Ensure you have Jupyter Notebook installed. If not, install it using:
   ```bash
   pip install notebook
## Steps to Run the Code in Jupyter


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


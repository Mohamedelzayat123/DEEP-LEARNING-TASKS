
# Linear Regression with PyTorch - Lab 1 Assignment

## Objective
The purpose of this assignment is to perform linear regression using an Artificial Neural Network (ANN) implemented in PyTorch. The project aims to analyze the relationship between property prices and their area, build an ANN model, train and evaluate it, and visualize the results. The target performance metric is an R-squared (R²) value of at least 0.75.

---

## Dataset Description
The dataset contains the following columns:

1. **Price**: The price of the property (target variable for regression).
2. **Area**: The size of the property in square meters.
3. **Bedrooms**: The number of bedrooms in the property.
4. **Bathrooms**: The number of bathrooms in the property.
5. **Stories**: The number of floors (stories) in the property.
6. **Mainroad**: Whether the property is adjacent to the main road (`yes` or `no`).
7. **Guestroom**: Whether the property has a guest room (`yes` or `no`).
8. **Basement**: Whether the property has a basement (`yes` or `no`).
9. **Hotwaterheating**: Whether the property is equipped with hot water heating (`yes` or `no`).
10. **Airconditioning**: Whether the property has air conditioning (`yes` or `no`).
11. **Parking**: The number of parking spaces available.
12. **Prefarea**: Whether the property is in a preferred area (`yes` or `no`).
13. **Furnishingstatus**: The furnishing status of the property (`furnished`, `semi-furnished`, or `unfurnished`).

The dataset may require preprocessing steps like encoding categorical variables, normalizing numerical features, and handling missing values to ensure accurate and reliable model predictions.

---

## Steps to Run the Code in Google Colab

### 1. Clone the GitHub Repository
Start by cloning the repository that contains the project files:
```bash
git clone <repository_link>
```

### 2. Open the `.ipynb` File
- Navigate to your Google Colab environment.
- Upload the `.ipynb` file provided in this repository.

### 3. Upload the Dataset
Ensure the dataset (`prices.csv` or equivalent file) is uploaded to your Colab environment. Modify the file path in the code if necessary.

### 4. Install Dependencies
Run the following command in Colab to install the required Python packages:
```bash
!pip install torch torchvision pandas matplotlib
```

### 5. Execute the Code
Run the cells sequentially in the notebook to:
- Load and preprocess the dataset.
- Train the ANN model.
- Evaluate the performance using metrics like MSE, MAE, and R².
- Visualize the predicted vs. actual prices.

---

## Dependencies
The following dependencies are required to run the code:
- Python 3.7+
- PyTorch 1.10+
- Pandas 1.3+
- Matplotlib 3.4+

You can install these using:
```bash
pip install torch pandas matplotlib
```

---

## Outputs
### Expected Deliverables
1. **Metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R-squared (R²)
   
2. **Visualizations**:
   - Graph comparing actual prices vs. predicted prices.

3. **Files**:
   - `.ipynb` notebook with the complete implementation.
   - Plot images as `.png` or `.jpeg`.

---

## Notes
- The model's architecture and hyperparameters can be adjusted to improve performance.
- Ensure to preprocess the dataset properly, including encoding categorical variables and normalizing numerical features.

Feel free to reach out if you encounter any issues running the code or have suggestions for improvement.

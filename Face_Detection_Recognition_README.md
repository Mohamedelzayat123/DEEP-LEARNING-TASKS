
# Face Detection & Recognition Using CNNs and Transfer Learning

## Objective
The objective of this project was to develop a real-time face detection and recognition system using deep learning techniques and transfer learning. The system successfully performs the following:

1. **Face Detection**: Detect faces in images using OpenCV’s deep learning-based detector.
2. **Face Recognition**: Recognize faces using a fine-tuned transfer learning model.
3. Achieved an accuracy of over **85%** on the validation dataset.
4. Implemented an additional "not identified" class for recognizing unclassified faces.
5. Integrated the model with a live camera feed for real-time operations.

---

## Dataset
The dataset used for this project was the **Pins Face Recognition Dataset**, available on Kaggle:
[https://www.kaggle.com/datasets/hereisburak/pins-face-recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition)

### Dataset Preparation
1. **Face Detection and Cropping**:
   - OpenCV’s DNN module was used to detect and crop faces in the dataset.
   - Cropped faces were saved into a new directory for better training performance.

2. **Data Augmentation**:
   - Augmented the dataset using rotation, flipping, and scaling to increase model robustness.

---

## Project Workflow
The project was divided into two main phases: Face Detection and Face Recognition.

### Phase 1: Face Detection
- **Implementation**:
  1. Used OpenCV’s pre-trained face detection model.
  2. Detected faces in all images from the dataset.
  3. Cropped and saved the faces for the recognition phase.

### Phase 2: Face Recognition
- **Implementation**:
  1. **Transfer Learning**:
     - Used ResNet50 as the pre-trained model.
     - Fine-tuned the model on the cropped face dataset.
     - Added additional dense layers and a softmax layer for classification.
  2. **Regularization**:
     - Applied dropout and L2 regularization to improve generalization.
  3. **Validation**:
     - Achieved over **88% accuracy** on the validation dataset.

### Recognizing Unclassified Faces
- Implemented cosine similarity to classify unrecognized faces as "not identified."

### Real-Time Detection and Recognition
- Integrated the model with a live camera feed to perform both face detection and recognition in real-time.

---

## Implementation Details

### Libraries and Tools
- **Deep Learning Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Data Manipulation**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

### Model Architecture
1. Pre-trained ResNet50 architecture was used as the base.
2. Custom dense layers and a softmax layer were added to tailor the model for the dataset.
3. Transfer learning was applied by freezing early layers and fine-tuning later layers.

### Regularization Techniques
1. Dropout layers to reduce overfitting.
2. L2 regularization for weight constraints.

### Metrics
Evaluated the model using:
1. **Accuracy**: Achieved over **88%** on the validation dataset.
2. **Mean Squared Error (MSE)**: Measured prediction errors.
3. **Mean Absolute Error (MAE)**: Analyzed deviations.
4. **R-squared (R²)**: Measured the goodness of fit.

---

## Steps to Run the Project

### 1. Clone the Repository
Clone the GitHub repository containing the project files:
```bash
git clone <repository_link>
```

### 2. Install Dependencies
Install the required Python libraries:
```bash
pip install tensorflow keras opencv-python matplotlib numpy pandas
```

### 3. Dataset Setup
1. Download the dataset from Kaggle and place it in the `data` directory.
2. Run the preprocessing script to detect and crop faces.

### 4. Train the Model
1. Open the provided Jupyter notebook or `.ipynb` file.
2. Train the model using the cropped face dataset.

### 5. Test Real-Time Recognition
1. Connect a webcam to your system.
2. Run the live recognition script to test real-time face detection and recognition.

---

## Deliverables
1. **GitHub Repository**:
   - Complete code with documentation.
   - Preprocessing scripts for cropping faces.
   - Training and validation notebooks.

2. **Performance Metrics**:
   - Accuracy: Over **88%**
   - MSE, MAE, and R² scores.

3. **Visualizations**:
   - Training and validation loss/accuracy plots.
   - Graphs comparing actual and predicted labels.

4. **Testing Video**:
   - Demonstration of the system recognizing faces:
     - Your face as "not identified."
     - Printed images of famous individuals (e.g., Elon Musk, Barack Obama).

---

## Notes
- ResNet50 provided robust performance for face recognition tasks.
- Regularization techniques significantly improved model generalization.
- Real-time testing validated the system’s efficiency and accuracy.

Feel free to reach out for assistance or to report any issues.

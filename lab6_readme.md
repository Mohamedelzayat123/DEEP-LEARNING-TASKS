
# Image Caption Generator

This repository contains a deep learning model for generating captions for images. The model uses the InceptionV3 architecture for image feature extraction and an LSTM-based model for generating the captions.


## Colab Link

You can run this project on Google Colab:

[Google Colab](https://colab.research.google.com/drive/1ZtDVy4gr9fwWz6Uqfm2_IWu4unl0lYVm?usp=sharing)



## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV
- tqdm
- Google Colab (optional for mounting Google Drive)



## Setup Instructions

### 1. Mount Google Drive (Optional)

If you are working on Google Colab, you can mount your Google Drive to access your dataset.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Loading the Pretrained InceptionV3 Model

The model used in this notebook is the InceptionV3 model from TensorFlow, pre-trained on ImageNet.

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

# Load the InceptionV3 model
model = InceptionV3()

# Re-structure the model to use the second-to-last layer for feature extraction
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Summarize the model architecture
print(model.summary())
```

### 3. Extracting Image Features

The notebook extracts features from images by loading them, resizing, and applying necessary preprocessing using the InceptionV3 model.

```python
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm.notebook import tqdm

features = {}
directory = '/content/drive/MyDrive/DEEP_lab6/Images'

for img_name in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, img_name)
    image = load_img(img_path, target_size=(299, 299))  # Resize the image
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Reshape
    image = preprocess_input(image)  # Preprocess the image
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

# Store the extracted features
import pickle
pickle.dump(features, open('features.pkl', 'wb'))
```

### 4. Loading Features from Pickle

Once the features are stored in a pickle file, they can be loaded for further processing.

```python
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)
```

### 5. Loading Captions

The captions for each image are read and mapped to their corresponding image IDs. The caption data is then preprocessed for model training.

```python
with open('captions.txt') as f:
    next(f)  # Skip header if exists
    captions_doc = f.read()

mapping = {}
for line in tqdm(captions_doc.split('
')):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

print(f'Number of image-caption pairs: {len(mapping)}')
```

### 6. Preprocessing Text Data

The captions are tokenized and padded to ensure uniform input size for the model.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

# Pad sequences to the same length
max_sequence_length = 34
sequences = tokenizer.texts_to_sequences(captions)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
```

### 7. Model Training and Real-Time Implementation

This section will cover the implementation of an LSTM model for training the caption generation model, and possibly an API or a stream to generate captions for new images in real-time.

## Acknowledgements

This project uses the InceptionV3 model from TensorFlow for feature extraction, and Keras for building and training the caption generation model.



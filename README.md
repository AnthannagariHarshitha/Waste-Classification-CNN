# Waste-Classification-CNN

## Overview

This project implements a Convolutional Neural Network (CNN) model to classify images of waste into two categories: Organic and Recyclable. The model is built using TensorFlow and Keras and leverages OpenCV for image processing.

## Project Structure

- `Dataset/` - Contains the images for training and testing in separate folders (`TRAIN` and `TEST`).
- `waste_management_cnn.py` - The main Python script that trains the CNN model.
- `README.md` - The file you are reading right now.

## Installation

Ensure that Python is installed on your machine, and use the following commands to install the necessary dependencies.

### 1. Install the required packages

```bash
pip install opencv-python
pip install tensorflow
pip install matplotlib
pip install pandas
pip install tqdm
```

### 2. Download the Dataset

Make sure you have a dataset of images organized in two categories:
- `TRAIN/Organic/`
- `TRAIN/Recyclable/`
- `TEST/Organic/`
- `TEST/Recyclable/`

You can place your images under these folders. The dataset should contain labeled images of organic and recyclable waste.

## Usage

### 1. Image Preprocessing

The code reads images from the provided `Dataset` and processes them using OpenCV. Each image is converted to RGB format and stored in `x_data`, while the corresponding labels are stored in `y_data`.

```python
x_data=[]
y_data=[]
for category in glob(train_path+'/*'):
    for file in tqdm(glob(category+'/*')):
        img_array = cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split('/')[-1])
```

### 2. Data Visualization

The script visualizes the distribution of the dataset by plotting a pie chart showing the percentage of images in the `Organic` and `Recyclable` categories.

```python
colors = ['#a0d157','#c48bb8']
plt.pie(data.label.value_counts(), labels=['Organic', 'Recyclable'], autopct='%0.2f%%', colors=colors, startangle=45)
plt.show()
```

### 3. CNN Model Architecture

The CNN model is constructed using the following layers:
- Conv2D layer for feature extraction.
- MaxPooling2D for downsampling.
- Dropout for regularization.
- Dense layers for classification.

The model is compiled with an Adam optimizer and categorical cross-entropy loss for multi-class classification.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 4. Training the Model

The model is trained using the `ImageDataGenerator` class for image augmentation to increase the diversity of the dataset. The images are resized to a consistent shape before being fed into the network.

```python
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

model.fit(train_generator, epochs=10, validation_data=validation_generator)
```

### 5. Evaluation and Prediction

Once the model is trained, you can evaluate its performance using the test dataset and make predictions on new images.

## Future Improvements

- Increase dataset size for better model performance.
- Experiment with other architectures (e.g., VGG16, ResNet).
- Add advanced image augmentation techniques.
- Implement real-time waste classification using a webcam.


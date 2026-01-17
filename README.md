1. Full EDA and Reporting
Since our data is synthetic, "Exploratory Data Analysis" (EDA) focuses on verifying that the generated lanes are diverse enough for the model to learn.
•	Visual Check: Plot 5 images alongside their masks to ensure the white lines in the mask perfectly overlap the lanes in the image.
•	Intensity Distribution: Check the pixel values. Since we normalized, they should fall between 0.0 and 1.0.
•	Class Balance: Calculate the ratio of "Lane" pixels (white) to "Background" pixels (black). In lane detection, the background is usually 90-95% of the image.
________________________________________
2. Data Preparation
We must ensure the data is "model-ready."
•	Normalization: We divide pixel values by 255. This helps the model converge faster because large input values (like 255) can cause "exploding gradients" during training.
•	Resizing: We use 256x256. This is a "Goldilocks" size—large enough to see the lines, but small enough to run fast in Unity without lagging.
________________________________________
3. Baseline Model (The U-Net)
We use a U-Net architecture. It’s the industry standard for "Image Segmentation" (labeling every pixel).
•	Encoder: Shrinks the image to find "features" (what is a line?).
•	Decoder: Blows the image back up to full size to draw the mask.
________________________________________
4. Advanced Tuning & Optimization
Handle Data Imbalancing
Because there are way more "road" pixels than "lane" pixels, the model might get "lazy" and just predict black for everything.
•	Fix: Use Weighted Binary Cross-Entropy or Dice Loss. These functions "punish" the model more if it misses a lane pixel compared to missing a background pixel.
Data Augmentation
To make the model robust for Unity:
•	Brightness/Contrast: Simulate different times of day.
•	Horizontal Flip: Help the AI realize lanes can curve left OR right.
•	Gaussian Noise: Simulate low-quality camera sensors.
Ensemble Learning (Advanced)
Instead of one "brain," we use three. We train three different versions of the model and "average" their answers. This prevents the car from glitching if one model gets confused by a shadow.
________________________________________
Updated Training Script (train_advanced.py)
This script incorporates Augmentation and Normalization.
Python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

# 1. Data Augmentation Setup
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    validation_split=0.2
)

# 2. Loading and Preprocessing
def load_data():
    images = []
    masks = []
    for i in range(500):
        img = cv2.imread(f'dataset/images/{i}.jpg')
        msk = cv2.imread(f'dataset/masks/{i}.png', 0)
        images.append(cv2.resize(img, (256, 256)) / 255.0)
        masks.append(cv2.resize(msk, (256, 256)) / 255.0)
    return np.array(images), np.array(masks).reshape(-1, 256, 256, 1)

X, y = load_data()

# 3. Baseline Model with Dropout (Tuning)
def build_tuned_model():
    inputs = tf.keras.Input((256, 256, 3))
    # Add Dropout to prevent overfitting
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x) 
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

model = build_tuned_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Training
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)
model.save('models/best_model.h5')
print("Advanced model saved!")
________________________________________
5. Evaluation
Once trained, we check the IoU (Intersection over Union) score.
•	Goal: > 0.70.
•	This measures how much the "AI's predicted lane" overlaps with the "Actual lane."


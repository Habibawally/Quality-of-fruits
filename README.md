🍎 Fruit Quality Classification using InceptionResNetV2
📘 Overview

This project focuses on classifying fruit quality (Good / Bad / Mixed) using a Convolutional Neural Network (CNN) based on the InceptionResNetV2 architecture.
The model is trained to distinguish between different fruit types and their quality levels using processed image data.

🚀 Features

Image classification using Transfer Learning (InceptionResNetV2).

Data preprocessing and augmentation for better generalization.

High accuracy (~96%) on the test dataset.

Visualizations for class distribution, accuracy, loss, and confusion matrix.

Saves the trained model as model_fruitNet.h5.

📂 Dataset

The dataset is structured as follows:

/Processed Images_Fruits
 ├── Bad Quality_Fruits
 │    ├── Apple
 │    ├── Banana
 │    └── ...
 ├── Good Quality_Fruits
 │    ├── Apple
 │    ├── Banana
 │    └── ...
 └── Mixed Quality_Fruits
      ├── Apple
      ├── Orange
      └── ...


Each folder contains fruit images labeled according to their quality.
Images are read and processed into a Pandas DataFrame with two columns:

filepaths – image file path

labels – fruit type and quality (e.g., Apple_Good_Good, Banana_Bad_Bad, Orange_mixed)

🧠 Model Architecture

The model uses InceptionResNetV2 (pretrained on ImageNet) as the convolutional base, with custom dense layers on top.

x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
output = Dense(18, activation='softmax')(x)


Optimizer: Adam (lr = 0.0001)
Loss Function: Categorical Cross-Entropy
Metrics: Accuracy

🧩 Training Details

Training Data: 80%

Validation Data: 20%

Early Stopping: Enabled (based on validation loss)

Epochs: 10

Training Accuracy: ~95.5%
Validation Accuracy: ~95.7%
Test Accuracy: ~96.2%

📊 Results & Evaluation

Accuracy and Loss Curves:

Training and validation accuracy increased steadily.

Loss decreased significantly across epochs.

Confusion Matrix:
Shows strong performance across all 18 fruit quality classes.

Accuracy Score: 0.9622


Example of Correct Predictions:
Displays a few test images correctly classified by the model.

🧰 Technologies Used

Python 🐍

TensorFlow / Keras

NumPy, Pandas

OpenCV

Matplotlib, Seaborn

Scikit-learn

💾 Saved Model

After training, the model is saved as:

model.save("model_fruitNet.h5")


You can later load and use it for predictions on new fruit images.

▶️ Run on Google Colab

You can open and run this notebook on Google Colab by clicking below:


📈 Future Improvements

Fine-tune InceptionResNetV2 layers for better performance.

Expand dataset with more fruit categories.

Deploy the model as a web or mobile app for real-time classification.

👩‍💻 Author

Habiba Wally
📍 AI & Computer Science Student | Deep Learning Enthusiast
💡 Passionate about building intelligent computer vision systems.

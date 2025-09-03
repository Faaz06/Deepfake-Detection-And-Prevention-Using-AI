# Deepfake-Detection-And-Prevention-Using-AI
AI-powered Deepfake Detection and Prevention system using Python, TensorFlow, and OpenCV.
# Deepfake Detection & Prevention

> A step-by-step guide to download, run, and get final results from the Deepfake Detection and Prevention repository.
# Deepfake Detection and Prevention

## Table of Contents
- Introduction  
- Documentation  
- Code  
- Usage  
- Training Details  
- Result Snapshots  
- Tools  

---

## Introduction
Welcome to my Deepfake Detection and Prevention project! In this comprehensive approach, I utilize advanced AI techniques to tackle the growing challenge of deepfake images. By leveraging machine learning algorithms and neural networks, this project aims to identify and mitigate the impact of manipulated images. You'll find detailed documentation, code implementations, and datasets used to train and test our models. Whether you're a researcher, developer, or just curious about AI's role in combating digital misinformation, this repository offers valuable insights and tools to understand and counter deepfakes effectively.

---

## Documentation
Check out the documentation of the project by clicking here!..  https://github.com/Faaz06/Deepfake-Detection-And-Prevention-Using-AI/blob/main/Deepfake%20detection%20and%20prevention%20Black%20book%20Documentation.pdf

---

## Code
Check out the code of the project by clicking here!

---

## Usage

### 1. Clone the Repository
git clone https://github.com/faazkhan/deepfake-detection.git

cd deepfake-detection

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to keep dependencies isolated.

python -m venv env

Activate the environment:  
**Windows:**
- .\env\Scripts\activate

**Linux/Mac:**
- source env/bin/activate

### 3. Install Dependencies
Install all required Python libraries from requirements.txt.
- pip install -r requirements.txt

### 4. Download the Dataset
- Download the dataset from Kaggle.  
- Place it inside the `dataset/` folder of the project.  

### 5. Train the Model
Run the training script to train the deepfake detection model:

 -python train.py
 Once training is complete, a `.h5` file (trained model) will be automatically saved in the project folder.

### 6. Run the Application
Run the Streamlit web app:

-streamlit run app.py


### 7. Open in Browser
After running, a new tab will open in your browser with the project interface.

### 8. Upload and Analyze Images
- Upload an image you want to test.  
- The model will process it and return a result: **Real** or **Fake**, along with reasoning.  

---

## Training Details
The model has been trained with the following configuration:
- **Model Architecture:** Convolutional Neural Networks (CNN) with transfer learning (MobileNetV2 / VGG16 depending on configuration).  
- **Optimizer:** Adam Optimizer (`learning_rate=0.001`)  
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy, Precision, Recall  
- **Epochs:** 20–30 (configurable based on system resources)  
- **Batch Size:** 32  
- **Dataset Split:** 80% training, 20% validation  
- **Output File:** A trained model is saved as `deepfake_model.h5` in the project folder.  

You can modify these hyperparameters in the training script (`train.py`) if needed.  

---

## Result Snapshots
- **Home Screen:** Displays introduction and project goal.  
- **Upload Image:** Allows users to upload images for testing.  
- **Real Output:** If the image is genuine, output will show **Real**.  
- **Fake Output:** If manipulated, output will show **Fake**.  

---

## Tools
- **Python** (TensorFlow, Keras, scikit-learn, OpenCV)  
- **Streamlit** (for deployment)  
- **PyCharm** (development environment)  
- **Kaggle Dataset**  

---

## Thanks for Watching
Thank you for checking out my Deepfake Detection and Prevention project! If you found this project useful or interesting, please consider giving it a star 🌟 on GitHub. Your support is greatly appreciated!




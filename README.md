# 🌱 Plant Disease Detection using YOLOv8


## 📌 Overview

This project focuses on detecting and localizing plant diseases using Object Detection techniques with YOLOv8.

The system analyzes plant images, detects infected regions, classifies disease types, and estimates disease severity.

Farmers can get help by only giving the model the plants' photos and wait for it telling them  what kind of diseases the plants suffer from and level of severity the infection is .

---

# 🎯 Objectives

* Detect diseased regions on plants
* Draw bounding boxes around infected areas
* Classify disease types
* Estimate disease severity and counts
* Compare model performance with and without data augmentation

---

# 🧠 Technologies Used

* Python
* YOLOv8
* OpenCV
* Ultralytics
* Computer Vision
* Deep Learning

---

# 📂 Dataset

The dataset used in this project can be accessed here:

https://universe.roboflow.com/fyp-advun/crop-disease-2rilx/dataset/1

The dataset consists of :
-23 classes 
-


The dataset contains images of healthy and diseased plants .

Each image includes:

* Bounding box annotations
* Disease class labels

Dataset split (after balancing):

* 79% Training
* 12% Evaluation
* 9%  Testing

---

# 🏷️ Annotation

Annotation was performed using tools such as:

* Roboflow

---

# ⚙️ Methodology

## 1. Data Preparation

* Image cleaning
* Annotation verification
* Dataset balancing including :
- Merging training , validation and testing subsets
- Duplication of the 18 minor classes 
* Dataset Resplitting

## 2. Model Training

YOLOv8 was used for object detection training.

Training parameters included:

* Epochs
* Batch size
* Image size

## 3. Experiments

Two main experiments were conducted:

* Training without augmentation
* Training with augmentation
* parameters tuning


Augmentation techniques included:

* Flip
* Rotation
* Zoom
* Brightness adjustment

---

# 📊 Evaluation Metrics

The model was evaluated using:

* Precision
* Recall
* mAP
* F1 score
* Confusion Matrix

---

# 📈 Results

The model successfully:

* Detected infected regions
* Classified diseases
* Generated confidence scores

Additional logic was implemented for:

* Counting infected regions
* Severity estimation

Severity levels:

* Mild (<10%)
* Moderate (10–30%)
* Severe (>30%)

---

# 🔥 Key Insight

Interestingly, the augmented model achieved better generalization on unseen data, despite having slightly-lower training performance compared to the non-augmented model.

This demonstrates the importance of data augmentation in improving model robustness and reducing overfitting.

---

# 🚀 Future Work

* Adding new photos to the minor classes rather than duplicating them
* Train the model on many-object-photos to learn detecting overlapped objects
* Use segmentation instead of bounding boxes
* Deploy as a web application
* Real-time mobile detection

---

# 👥 Work Distribution

* Dataset & Annotation
* Model Training
* Experimentation & Optimization
* Evaluation & Visualization
* Logic Implementation & Presentation

---

# 📌 Conclusion

This project provided practical experience in object detection, model optimization, dataset preparation, and real-world AI problem solving using YOLOv8.
This project will serve people working in the farming field to know different kinds of plant diseases and pay attention for crops that suffer of severe infection.

---

# Author
Renad Amr 

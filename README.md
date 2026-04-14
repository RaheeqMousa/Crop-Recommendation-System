# Crop Recommendation System using Decision Tree

## Team
- Raheeq Mousa
- Heba Jamal

## Overview
This project uses a Decision Tree classifier to recommend the most suitable crop based on environmental and soil conditions such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

## Objective
To build a machine learning model that predicts the best crop for given environmental conditions to support agricultural decision-making.

## Dataset
- Source: Kaggle Crop Recommendation Dataset  
- Features:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall  
- Target:
  - Crop label (e.g., rice, maize, mango)

## Results
- Accuracy: ~98.7%
- Precision: ~98.9%
- Recall: ~98.7%
The model shows strong and consistent performance across all folds.

## How to Run
```bash
pip install pandas scikit-learn matplotlib
python main.py

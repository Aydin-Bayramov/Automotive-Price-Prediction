# Automotive Price Prediction Project

## Overview
This project provides a comprehensive, end-to-end pipeline for predicting car prices using machine learning. It automates the process of web scraping car listings from Turbo.az, preprocessing and cleaning the data, training multiple regression models, and making accurate price predictions. The solution leverages modern data science tools such as Selenium, Pandas, Scikit-Learn, and advanced machine learning techniques.

### Key Features
- **Web Scraping**: Uses Selenium to dynamically extract car listings (brands such as Mercedes, Hyundai, Kia, BMW) and maintains structured logs.
- **Data Preprocessing**: Cleans and transforms data by handling missing values, applying label and one-hot encoding, and scaling numerical features.
- **Model Training & Evaluation**: Implements multiple regression models (Linear Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and Stacking Regressor) with hyperparameter tuning via GridSearchCV.
- **Prediction Interface**: Provides a streamlined process to predict car prices based on input features.
- **Modular Architecture**: Well-organized codebase with separate modules for scraping, preprocessing, training, and prediction.

## Project Workflow
1. **Data Scraping** – Extracts car listing details from Turbo.az.
2. **Data Preprocessing** – Cleans, filters, and encodes data to prepare it for modeling.
3. **Model Training** – Trains and evaluates multiple regression models.
4. **Prediction** – Uses a trained stacking regressor to predict car prices.

---

## Model Performance

### Stacking Regression Results
#### Training Set:
- **Mean Absolute Error (MAE):** 3,308.25
- **Mean Squared Error (MSE):** 37,392,741.10
- **Root Mean Squared Error (RMSE):** 6,114.96
- **R² Score:** 0.9924

#### Test Set:
- **Mean Absolute Error (MAE):** 6,714.14
- **Mean Squared Error (MSE):** 121,101,077.96
- **Root Mean Squared Error (RMSE):** 11,004.59
- **R² Score:** 0.9751

---

## Directory Structure
```bash
├── data
│   ├── raw                   # Raw scraped data
│   ├── interim               # Intermediate cleaned data
│   └── processed             # Final data ready for modeling
├── models
│   ├── preprocessing_objects  # Encoders and scalers
│   └── trained_models         # Trained ML models
├── notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── Model_Turbo_AZ.ipynb  # Model training and evaluation
│   ├── Preprocessing.ipynb   # Data preprocessing workflow
│   └── Scraping.ipynb        # Data scraping process
├── src
│   ├── data_preprocessing
│   │   └── preprocessing.py    # Data processing script
│   ├── models
│   │   ├── predict.py          # Prediction script
│   │   └── train.py            # Model training script
│   └── web_scraping
│       └── scraping.py         # Web scraping script
├── logs
│   └── scraping.log            # Log file for web scraping
└── requirements.txt            # Project dependencies
```

---

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/Aydin-Bayramov/Automotive-Price-Prediction.git
cd Automotive-Price-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide
### 1. Data Scraping
Extract car listings from Turbo.az:
```bash
python scraping.py
```
- Scrapes car details (price, make, model, year, etc.) and saves them as CSV files in `data/raw/`.

### 2. Data Preprocessing
Clean and transform the scraped data:
```bash
python preprocessing.py
```
- Loads raw data, filters key models, processes missing values, applies encoding, and saves processed data in `data/processed/`.

### 3. Model Training
Train and evaluate machine learning models:
```bash
python train.py
```
- Splits data into train/test sets, tunes hyperparameters, trains models, and saves the best ones in `models/trained_models/`.
- A final **Stacking Regressor** is trained using the best-performing models.

### 4. Price Prediction
Predict car prices using the trained model:
```bash
python predict.py
```
- Loads the trained stacking regressor.
- Processes user input to match training format.
- Outputs the predicted car price in AZN.

---

## Contribution & Development
Contributions are highly encouraged! Feel free to submit issues or pull requests to improve the project.


# 💻 Laptop-Price-Predictor
An ML-powered web app that estimates a laptop's price from its specifications —built with XGBoost, scikit-learn, and Streamlit.

## 📖 Overview

Laptop prices vary wildly even for similar-looking specs, making it hard to knowif you're paying a fair price. This app uses a machine learning model trained onreal laptop data (~1,300 laptops) to predict the market price of a laptop basedon its brand, CPU, GPU, RAM, storage, display, and OS.

Enter the specs → get an instant price estimate in ₹.

## 🖼️ Screenshots
![Dashboard Preview](screenshot1.png)
![Dashboard Preview](screenshot2.png)

##✨ Features
1.🏷️ Select Company, Laptop Type, CPU brand, GPU brand, and OS from dropdowns
2.⚙️ Configure RAM, Weight, Screen Size (inches), Resolution (PPI), Touchscreen & IPS display
3.💾 Choose HDD / SSD storage combinations
4.⚡ Instant price prediction using a tuned XGBoost model
5.📊 Model trained with log-transformed target for better accuracy on a wide price range

## 🛠️ Tech Stack

Language - Python 3.x
ML Model - XGBoost (XGBRegressor), Random Forest, Adabost, GraradientBoost, LinearRegressor, VotingRegressor.
Preprocessing - scikit-learn (OneHotEncoder, ColumnTransformer, Pipeline)
Web App - Streamlit
Data Handling -	pandas, NumPy
Serialization - pickle

## 🧠 Model & Approach

### Preprocessing

One-Hot Encoding of categorical features: Company, TypeName, CPU brand, GPU brand, OS
Numerical features (RAM, Weight, PPI, HDD, SSD, etc.) passed through
Target (Price) log-transformed to handle skew, inverse-transformed at prediction time
Wrapped in a single scikit-learn Pipeline → preprocessing + model in one artifact

## Model

Final model - RandomForestRegressor(n_estimators=300,random_state=42,max_samples=0.63,max_features=0.64,max_depth=13)
Model | R² Score | MAE
RandomForestRegressor | 0.888 | 1.169
ExtraTreesRegressor | 0.8848 | 1.174
GradientBoostingRegressor | 0.8948 | 1.164
XGBoostRegressor |  0.902 | 1.1538
AdaBoostRegressor | 0.8488 | 1.2168
VotingRegressor(RandomForest,GradientBoost,XGBoost) | 0.9037 | 1.156
LinearRegressor | 0.8070 | 1.233
Ridge | 0.812 | 1.233

## 🚀 Getting Started

### Prerequisites
Python 3.9+

### Installation
Clone the repositorygit clone https://github.com/[your-username]/Laptop-Price-Predictor.gitcd laptop-price-predictor Install dependencies
```bash
pip install -r streamlit_app/requirements.txt
python -m streamlit run app.py
```
The app will open at http://localhost:8501.

## 📊 Dataset
Source: 🔗 [laptopPriceDataset](https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset)
Size: ~1,300 laptops
Features: Company, TypeName, RAM (GB), Weight (kg), Touchscreen, IPS Panel,Screen Resolution (PPI), CPU brand, HDD (GB), SSD (GB), GPU brand, OS

## 📁 Project Structure
```text
├── data_cleaning_model_training/           # Python cleaning & model training
│   ├── LaptopPricePredictor.ipynb
│
├── streamlit_app/              # run the streamlit app in python
│   ├── app.py
│   └── requirements.txt
│
└── app_screenshots/
│   ├── screenshot1.png
    ├── screenshot2.png        # Streamlit app screenshots
```

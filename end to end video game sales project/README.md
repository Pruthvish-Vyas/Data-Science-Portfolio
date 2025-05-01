# Video Game Sales Prediction - ML Project

## 📋 Overview
This project is an end-to-end machine learning application that predicts global video game sales based on regional sales data and game metadata. The system uses historical video game sales data to train a regression model that can forecast the global commercial performance of games.

## 🎮 Features
Sales Prediction: Predict global sales figures based on regional performance and game characteristics
Interactive UI: User-friendly Streamlit interface for making predictions
Data Visualization: Visual representation of sales distribution across regions
Modular Design: Well-structured codebase with separation of concerns
## 🛠️ Tech Stack
Python: Core programming language
Pandas & NumPy: Data manipulation and numerical operations
Scikit-learn: Machine learning algorithms and preprocessing
Streamlit: Interactive web application
Pickle: Model serialization

## 🏗️ Project Structure

```
end-to-end-project/
│
├── artifacts/                  # Saved model and preprocessor objects
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── src/                        # Source code
│   ├── components/
│   │   ├── data_ingestion.py   # Data loading and splitting
│   │   ├── data_transformation.py # Feature engineering and preprocessing
│   │   └── model_trainer.py    # Model training and evaluation
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py # Prediction pipeline and custom data handling
│   │
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Logging configuration
│   └── utils.py                # Utility functions
│
├── game_sales_app.py           # Streamlit application for game sales prediction
├── app.py                      # Example Streamlit application (student scores)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```




## 🚀 Installation & Usage

Prerequisites
- Python 3.7+
- pip



### Setup

1. clone the repository:

```
git clone https://github.com/yourusername/video-game-sales-prediction.git
cd video-game-sales-prediction
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the Streamlit application:
```
streamlit run "end to end project/game_sales_app.py"
```

## 📊 Data Features
The model uses the following features to predict global video game sales:

**Categorical Features:**
Name: Title of the video game
Platform: Gaming platform (PS4, Xbox, PC, etc.)
Genre: Game genre (Action, RPG, Sports, etc.)
Publisher: Company that published the game

**Numerical Features:**
Year: Release year of the game
NA_Sales: Sales in North America (in millions)
EU_Sales: Sales in Europe (in millions)
JP_Sales: Sales in Japan (in millions)
Other_Sales: Sales in other regions (in millions)

**Target Variable:**
Global_Sales: Total worldwide sales (in millions)

## 🔄 Machine Learning Pipeline

1. Data Ingestion: Load and split data into training and testing sets
2. Data Transformation:
- Handle missing values with median imputation for numerical features
- Handle missing values with most frequent imputation for categorical features
- One-hot encode categorical variables
- Scale all features using StandardScaler

3. Model Training: Train regression model on transformed data
4. Prediction: Process new data through the same pipeline for consistent predictions

## 📸 Demo


## 🔮 Future Improvements
- Add more advanced features like game review scores
- Implement more sophisticated models (ensemble methods, deep learning)
- Add time series analysis for trend forecasting
- Expand the UI with more visualization options

👥 Author- [Pruthvish Vyas](https://github.com/Pruthvish-Vyas)


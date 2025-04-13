import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging

# Title of the Streamlit app
st.title("Video Game Global Sales Prediction")
st.write("This app predicts global sales for video games based on various features.")

# Input fields for user data
st.sidebar.header("Input Features")

# Categorical Features
name = st.sidebar.text_input("Game Name", "New Game")
platform = st.sidebar.selectbox("Platform", ["PS4", "Xbox One", "PC", "Switch", "PS5", "Xbox Series X", "Mobile", "Other"])
genre = st.sidebar.selectbox("Genre", ["Action", "Adventure", "RPG", "Sports", "Shooter", "Strategy", "Puzzle", "Racing", "Simulation"])
publisher = st.sidebar.selectbox("Publisher", ["EA", "Ubisoft", "Activision", "Nintendo", "Sony", "Microsoft", "Take-Two", "Other"])

# Numerical Features
year = st.sidebar.slider("Release Year", 2000, 2023, 2022)
na_sales = st.sidebar.slider("North America Sales (millions)", 0.0, 10.0, 1.0, 0.1)
eu_sales = st.sidebar.slider("Europe Sales (millions)", 0.0, 10.0, 0.8, 0.1)
jp_sales = st.sidebar.slider("Japan Sales (millions)", 0.0, 5.0, 1.2, 0.1)
other_sales = st.sidebar.slider("Other Regions Sales (millions)", 0.0, 5.0, 0.5, 0.1)

# Button to make predictions
if st.button("Predict Global Sales"):
    try:
        # Create a CustomData instance with user inputs
        custom_data = CustomData(
            Name=name,
            Platform=platform,
            Genre=genre,
            Publisher=publisher,
            Year=year,
            NA_Sales=na_sales,
            EU_Sales=eu_sales,
            JP_Sales=jp_sales,
            Other_Sales=other_sales
        )

        # Convert the input data to a DataFrame
        input_data = custom_data.get_data_as_data_frame()
        
        # Display the input data
        st.subheader("Input Data")
        st.write(input_data)

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()

        # Make predictions
        with st.spinner('Calculating prediction...'):
            predictions = predict_pipeline.predict(input_data)

        # Display the prediction
        st.success(f"The predicted global sales is: {predictions[0]:.2f} million units")
        
        # Calculate and display the sum of regional sales for comparison
        regional_sum = na_sales + eu_sales + jp_sales + other_sales
        st.info(f"Sum of regional sales: {regional_sum:.2f} million units")
        
        # Display a simple bar chart of sales by region
        sales_data = {
            'Region': ['North America', 'Europe', 'Japan', 'Other', 'Predicted Global'],
            'Sales (millions)': [na_sales, eu_sales, jp_sales, other_sales, predictions[0]]
        }
        sales_df = pd.DataFrame(sales_data)
        st.subheader("Sales by Region")
        st.bar_chart(sales_df.set_index('Region'))

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"Prediction error: {str(e)}")

    logging.info('Prediction completed')

# Add some additional information about the model
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This application predicts global video game sales based on regional sales "
    "and game information. The model was trained on historical video game sales data."
)

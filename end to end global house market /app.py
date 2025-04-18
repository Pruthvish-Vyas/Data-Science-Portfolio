import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline, get_house_price_prediction

# Set page configuration
st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Global Housing Price Prediction")
st.markdown("""
This application predicts housing prices based on various economic and demographic factors.
Enter the details below to get a prediction for the House Price Index.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Data Exploration", "Model Information"])

# Load sample data for visualization
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('notebook/data/global_housing_market_extended.csv')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

# Prediction page
if page == "Prediction":
    st.header("Housing Price Prediction")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Input fields for prediction
        country = st.selectbox(
            "Country",
            options=data["Country"].unique().tolist() if data is not None else ["USA", "UK", "Canada", "Germany", "France"]
        )
        
        year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)
        
        rent_index = st.slider(
            "Rent Index", 
            min_value=50.0, 
            max_value=120.0, 
            value=80.0,
            help="Rent Index measures the relative cost of renting in a city or country"
        )
        
        affordability_ratio = st.slider(
            "Affordability Ratio", 
            min_value=3.0, 
            max_value=12.0, 
            value=7.0,
            help="Ratio of median house price to median annual household income"
        )
        
        mortgage_rate = st.slider(
            "Mortgage Rate (%)", 
            min_value=1.0, 
            max_value=7.0, 
            value=4.0,
            help="Average mortgage interest rate"
        )
    
    with col2:
        inflation_rate = st.slider(
            "Inflation Rate (%)", 
            min_value=0.5, 
            max_value=7.0, 
            value=3.0,
            help="Annual inflation rate"
        )
        
        gdp_growth = st.slider(
            "GDP Growth (%)", 
            min_value=-2.0, 
            max_value=6.0, 
            value=2.0,
            help="Annual GDP growth rate"
        )
        
        population_growth = st.slider(
            "Population Growth (%)", 
            min_value=-1.0, 
            max_value=2.5, 
            value=0.7,
            help="Annual population growth rate"
        )
        
        urbanization_rate = st.slider(
            "Urbanization Rate (%)", 
            min_value=60.0, 
            max_value=90.0, 
            value=75.0,
            help="Percentage of population living in urban areas"
        )
        
        construction_index = st.slider(
            "Construction Index", 
            min_value=70.0, 
            max_value=150.0, 
            value=110.0,
            help="Index measuring construction activity"
        )
    
    # Prediction button
    if st.button("Predict House Price Index"):
        try:
            with st.spinner("Calculating prediction..."):
                # Get prediction
                prediction = get_house_price_prediction(
                    country=country,
                    year=year,
                    rent_index=rent_index,
                    affordability_ratio=affordability_ratio,
                    mortgage_rate=mortgage_rate,
                    inflation_rate=inflation_rate,
                    gdp_growth=gdp_growth,
                    population_growth=population_growth,
                    urbanization_rate=urbanization_rate,
                    construction_index=construction_index
                )
            
            # Display prediction with nice formatting
            st.success(f"Predicted House Price Index: **{prediction:.2f}**")
            
            # Add some context to the prediction
            if data is not None:
                country_avg = data[data['Country'] == country]['House Price Index'].mean()
                overall_avg = data['House Price Index'].mean()
                
                st.info(f"""
                **Context:**
                - Average House Price Index for {country}: {country_avg:.2f}
                - Overall average House Price Index: {overall_avg:.2f}
                """)
                
                # Create a gauge chart to visualize where the prediction falls
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Define the range
                min_val = data['House Price Index'].min()
                max_val = data['House Price Index'].max()
                
                # Create a horizontal bar for the range
                ax.barh([0], [max_val - min_val], left=[min_val], height=0.4, color='lightgray')
                
                # Add the prediction point
                ax.scatter([prediction], [0], color='red', s=100, zorder=5)
                
                # Add country average
                ax.scatter([country_avg], [0], color='blue', s=100, zorder=5, alpha=0.7)
                
                # Add labels
                ax.text(prediction, 0.1, f"Prediction: {prediction:.2f}", ha='center', va='bottom', color='red')
                ax.text(country_avg, -0.1, f"{country} Avg: {country_avg:.2f}", ha='center', va='top', color='blue')
                
                # Remove y-axis and set x-axis limits
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlim(min_val - 5, max_val + 5)
                ax.set_yticks([])
                ax.set_title("Prediction vs. Range of House Price Index")
                
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check your inputs and try again.")

# Data Exploration page
elif page == "Data Exploration":
    st.header("Data Exploration")
    
    if data is not None:
        # Show basic statistics
        st.subheader("Dataset Overview")
        st.write(f"Number of records: {data.shape[0]}")
        st.write(f"Number of countries: {data['Country'].nunique()}")
        st.write(f"Year range: {data['Year'].min()} to {data['Year'].max()}")
        
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(data.head())
        
        # Data visualization
        st.subheader("Data Visualization")
        
        viz_type = st.selectbox(
            "Select Visualization",
            ["House Price Index by Country", "Correlation Matrix", "House Price vs. Economic Factors", 
             "Time Series of House Prices", "Distribution of House Prices"]
        )
        
        if viz_type == "House Price Index by Country":
            fig, ax = plt.subplots(figsize=(12, 6))
            country_avg = data.groupby('Country')['House Price Index'].mean().sort_values(ascending=False)
            sns.barplot(x=country_avg.index, y=country_avg.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title("Average House Price Index by Country")
            ax.set_ylabel("House Price Index")
            st.pyplot(fig)
            
        elif viz_type == "Correlation Matrix":
            fig, ax = plt.subplots(figsize=(10, 8))
            numeric_data = data.select_dtypes(include=[np.number])
            corr = numeric_data.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title("Correlation Matrix of Numeric Features")
            st.pyplot(fig)
            
        elif viz_type == "House Price vs. Economic Factors":
            factor = st.selectbox(
                "Select Economic Factor",
                ["Rent Index", "Affordability Ratio", "Mortgage Rate (%)", 
                 "Inflation Rate (%)", "GDP Growth (%)", "Construction Index"]
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=factor, y="House Price Index", hue="Country", data=data, ax=ax)
            ax.set_title(f"House Price Index vs. {factor}")
            st.pyplot(fig)
            
        elif viz_type == "Time Series of House Prices":
            countries = st.multiselect(
                "Select Countries",
                options=data["Country"].unique().tolist(),
                default=["USA", "UK", "Canada"]
            )
            
            if countries:
                fig, ax = plt.subplots(figsize=(12, 6))
                for country in countries:
                    country_data = data[data["Country"] == country]
                    ax.plot(country_data["Year"], country_data["House Price Index"], marker='o', label=country)
                
                ax.set_title("House Price Index Over Time")
                ax.set_xlabel("Year")
                ax.set_ylabel("House Price Index")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.warning("Please select at least one country.")
                
        elif viz_type == "Distribution of House Prices":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data["House Price Index"], kde=True, ax=ax)
            ax.set_title("Distribution of House Price Index")
            ax.set_xlabel("House Price Index")
            
            # Add vertical lines for mean and median
            mean_val = data["House Price Index"].mean()
            median_val = data["House Price Index"].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.legend()
            
            st.pyplot(fig)

# Model Information page
elif page == "Model Information":
    st.header("Model Information")
    
    st.subheader("About the Model")
    st.write("""
    This application uses machine learning to predict the House Price Index based on various economic and demographic factors.
    
    **Features used for prediction:**
    - Country: The country for which the prediction is made
    - Year: The year for which the prediction is made
    - Rent Index: Measures the relative cost of renting
    - Affordability Ratio: Ratio of median house price to median annual household income
    - Mortgage Rate (%): Average mortgage interest rate
    - Inflation Rate (%): Annual inflation rate
    - GDP Growth (%): Annual GDP growth rate
    - Population Growth (%): Annual population growth rate
    - Urbanization Rate (%): Percentage of population living in urban areas
    - Construction Index: Index measuring construction activity
    
    **Model Type:**
    The application uses a regression model trained on historical housing market data. The model was selected after evaluating multiple algorithms including:
    - Linear Regression
    - Ridge Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    
    The best performing model was selected based on metrics such as R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
    """)
    
    st.subheader("Feature Importance")
    st.write("""
    While we don't display the exact feature importance values here, generally the most important factors for predicting house prices include:
    
    1. Rent Index - Higher rent typically correlates with higher house prices
    2. Affordability Ratio - Directly related to house prices
    3. Mortgage Rate - Lower rates often lead to higher house prices
    4. GDP Growth - Economic growth typically drives housing markets
    5. Construction Index - Supply of new housing affects prices
    
    The importance of each factor can vary by country and economic conditions.
    """)
    
    st.subheader("How to Use the Prediction Tool")
    st.write("""
    To get a prediction:
    1. Go to the Prediction page
    2. Enter values for all required fields
    3. Click the "Predict House Price Index" button
    4. View your prediction and contextual information
    
    You can adjust the values to see how different factors might affect housing prices.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Housing Price Prediction App | Created with Streamlit")

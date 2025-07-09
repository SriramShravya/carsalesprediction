import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Car Sales Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .suggestion-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚗 Car Sales Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Analysis", "Sales Prediction", "Model Performance"])

# Cache functions for better performance
@st.cache_data
def load_and_process_data():
    """Load and process the car sales data"""
    # Note: Replace with your actual data loading
    # For demo purposes, I'll create sample data structure
    # In your actual deployment, load from your CSV file
    
    # You would replace this with: df = pd.read_csv('salesofcar.csv')
    # For now, creating a sample structure based on your code
    st.info("Please upload your 'salesofcar.csv' file or modify the load_and_process_data() function to load your data.")
    return None, None, None, None, None

@st.cache_resource
def train_model(sales_data):
    """Train the XGBoost model"""
    if sales_data is None:
        return None, None, None, None
    
    # Prepare features and target
    X = sales_data.drop(columns=['Sales'])
    y = sales_data['Sales']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'alpha': 0.5,
        'seed': 42
    }
    
    # Train model
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals,
                      early_stopping_rounds=30, verbose_eval=False)
    
    # Evaluate
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, X.columns.tolist(), rmse

def suggest_sales_improvements(input_data, historical_df):
    """Generate sales improvement suggestions"""
    suggestions = []
    
    # Convert single-element arrays to scalars
    get_scalar = lambda x: x[0] if isinstance(x, (list, np.ndarray, pd.Series)) else x
    
    model_enc = get_scalar(input_data.get('Model_encoded'))
    month = get_scalar(input_data.get('Month'))
    year = get_scalar(input_data.get('Year'))
    price = get_scalar(input_data.get('Price ($)', input_data.get('Avg_Price', None)))
    
    # Price comparison
    model_prices = historical_df[historical_df['Model_encoded'] == model_enc]['Avg_Price']
    if price and not model_prices.empty and price > model_prices.mean():
        suggestions.append("💰 Consider reducing the price — it's above average for this model.")
    
    # Monthly trend
    model_month_sales = historical_df[historical_df['Model_encoded'] == model_enc]
    monthly_avg = model_month_sales.groupby('Month')['Sales'].mean()
    if month in monthly_avg.index and monthly_avg[month] < monthly_avg.mean():
        suggestions.append("📅 Sales are usually lower in this month — consider promotions or marketing.")
    
    # Year-over-year trend
    if year in [2022, 2023]:
        prev_year = year - 1
        this_year_avg = historical_df[(historical_df['Model_encoded'] == model_enc) & 
                                     (historical_df['Year'] == year)]['Sales'].mean()
        prev_year_avg = historical_df[(historical_df['Model_encoded'] == model_enc) & 
                                     (historical_df['Year'] == prev_year)]['Sales'].mean()
        if not np.isnan(prev_year_avg) and this_year_avg < prev_year_avg:
            suggestions.append("📉 Sales are declining year-over-year. Consider investigating the cause.")
    
    return suggestions

# File upload section
uploaded_file = st.file_uploader("Upload your car sales CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Data preprocessing (based on your code)
    with st.spinner("Processing data..."):
        # Fill missing values
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        # Drop unwanted columns
        columns_to_drop = ['Car_id', 'Phone', 'Dealer_No ']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        
        # Process date
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        
        # Label encoding
        label_encoders = {}
        
        # Gender
        le_gender = LabelEncoder()
        df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
        label_encoders['Gender'] = le_gender
        
        # Other categorical variables
        categorical_features = ['Dealer_Name', 'Company', 'Model', 'Engine', 'Transmission']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature])
                label_encoders[feature] = le
        
        # One-hot encoding for specific features
        if 'Color' in df.columns:
            color_dummies = pd.get_dummies(df['Color'], prefix='Color')
            df = pd.concat([df, color_dummies], axis=1)
        
        if 'Body Style' in df.columns:
            body_style_dummies = pd.get_dummies(df['Body Style'], prefix='BodyStyle')
            df = pd.concat([df, body_style_dummies], axis=1)
        
        if 'Dealer_Region' in df.columns:
            dealer_region_dummies = pd.get_dummies(df['Dealer_Region'], prefix='DealerRegion')
            df = pd.concat([df, dealer_region_dummies], axis=1)
        
        # Create sales data
        group_keys = ['Year', 'Month', 'Model_encoded']
        
        # Sales counts
        sales_counts = df.groupby(group_keys).size().reset_index(name='Sales')
        
        # Aggregate features
        cont_agg = df.groupby(group_keys).agg({
            'Price ($)': 'mean',
            'Annual Income': 'mean'
        }).rename(columns={
            'Price ($)': 'Avg_Price',
            'Annual Income': 'Avg_Annual_Income'
        }).reset_index()
        
        # Merge data
        sales_data = sales_counts.merge(cont_agg, on=group_keys)
        
        # Add other encoded features
        cat_cols = [col for col in df.columns if col.endswith('_encoded') and col not in group_keys]
        if cat_cols:
            cat_agg = df.groupby(group_keys)[cat_cols].agg(lambda s: s.mode()[0]).reset_index()
            sales_data = sales_data.merge(cat_agg, on=group_keys)
        
        # Add boolean columns
        bool_cols = [col for col in df.columns if df[col].dtype == 'bool' or col.startswith(('Color_', 'BodyStyle_', 'DealerRegion_'))]
        if bool_cols:
            bool_agg = df.groupby(group_keys)[bool_cols].max().reset_index()
            sales_data = sales_data.merge(bool_agg, on=group_keys)
    
    # Train model
    model, scaler, expected_columns, rmse = train_model(sales_data)
    
    # Page content based on selection
    if page == "Data Analysis":
        st.header("📊 Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Total Records:** {len(df)}")
            st.write(f"**Features:** {len(df.columns)}")
            st.write(f"**Date Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            
            st.subheader("Top 10 Companies by Sales")
            company_sales = df['Company'].value_counts().head(10)
            fig_company = px.bar(x=company_sales.index, y=company_sales.values, 
                               title="Top 10 Companies by Sales Count")
            st.plotly_chart(fig_company, use_container_width=True)
        
        with col2:
            st.subheader("Sales Distribution")
            fig_sales = px.histogram(sales_data, x='Sales', nbins=20, 
                                   title="Sales Distribution")
            st.plotly_chart(fig_sales, use_container_width=True)
            
            st.subheader("Monthly Sales Trend")
            monthly_sales = df.groupby('Month').size().reset_index(name='Sales')
            fig_monthly = px.line(monthly_sales, x='Month', y='Sales', 
                                title="Monthly Sales Trend")
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    elif page == "Sales Prediction":
        st.header("🔮 Sales Prediction")
        
        if model is not None:
            st.subheader("Enter Car Details for Prediction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                year = st.selectbox("Year", [2024, 2025, 2026], index=0)
                month = st.selectbox("Month", range(1, 13), index=0)
                model_encoded = st.selectbox("Model (encoded)", range(0, 50), index=0)
                avg_price = st.number_input("Average Price ($)", min_value=10000, max_value=100000, value=30000)
            
            with col2:
                avg_income = st.number_input("Average Annual Income ($)", min_value=20000, max_value=150000, value=55000)
                gender_encoded = st.selectbox("Gender (encoded)", [0, 1], index=0)
                company_encoded = st.selectbox("Company (encoded)", range(0, 20), index=0)
                engine_encoded = st.selectbox("Engine (encoded)", range(0, 5), index=0)
            
            with col3:
                transmission_encoded = st.selectbox("Transmission (encoded)", range(0, 3), index=0)
                dealer_name_encoded = st.selectbox("Dealer Name (encoded)", range(0, 50), index=0)
                
                # Color selection
                st.subheader("Color")
                color_black = st.checkbox("Black")
                color_white = st.checkbox("White")
                color_red = st.checkbox("Red")
            
            # Body Style and Region selections
            col4, col5 = st.columns(2)
            
            with col4:
                st.subheader("Body Style")
                body_hatchback = st.checkbox("Hatchback")
                body_suv = st.checkbox("SUV")
                body_sedan = st.checkbox("Sedan")
            
            with col5:
                st.subheader("Dealer Region")
                region_pasco = st.checkbox("Pasco")
                region_scottsdale = st.checkbox("Scottsdale")
                region_austin = st.checkbox("Austin")
            
            if st.button("Predict Sales", type="primary"):
                # Prepare input data
                test_input = {
                    'Year': [year],
                    'Month': [month],
                    'Model_encoded': [model_encoded],
                    'Avg_Price': [avg_price],
                    'Avg_Annual_Income': [avg_income],
                    'Gender_encoded': [gender_encoded],
                    'Company_encoded': [company_encoded],
                    'Engine_encoded': [engine_encoded],
                    'Transmission_encoded': [transmission_encoded],
                    'Dealer_Name_encoded': [dealer_name_encoded],
                    'Color_Black': [int(color_black)],
                    'Color_Pale White': [int(color_white)],
                    'Color_Red': [int(color_red)],
                    'BodyStyle_Hatchback': [int(body_hatchback)],
                    'BodyStyle_SUV': [int(body_suv)],
                    'BodyStyle_Sedan': [int(body_sedan)],
                    'DealerRegion_Pasco': [int(region_pasco)],
                    'DealerRegion_Scottsdale': [int(region_scottsdale)],
                    'DealerRegion_Austin': [int(region_austin)]
                }
                
                # Add missing columns with default values
                for col in expected_columns:
                    if col not in test_input:
                        test_input[col] = [0]
                
                # Make prediction
                try:
                    new_data = pd.DataFrame(test_input)
                    new_data = new_data[expected_columns]
                    scaled_data = scaler.transform(new_data)
                    dnew = xgb.DMatrix(scaled_data)
                    
                    prediction = model.predict(dnew)[0]
                    
                    # Display results
                    st.success(f"🎯 Predicted Sales: **{prediction:.2f}** units")
                    
                    # Generate suggestions
                    suggestions = suggest_sales_improvements(test_input, sales_data)
                    
                    if suggestions:
                        st.subheader("💡 Improvement Suggestions")
                        for suggestion in suggestions:
                            st.markdown(f'<div class="suggestion-box">{suggestion}</div>', 
                                      unsafe_allow_html=True)
                    else:
                        st.info("✅ No specific suggestions — input looks optimal!")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.error("Model not trained. Please check your data.")
    
    elif page == "Model Performance":
        st.header("📈 Model Performance")
        
        if model is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div class="metric-container"><h3>Model RMSE</h3><h2>{rmse:.4f}</h2></div>', 
                          unsafe_allow_html=True)
                
                # Feature importance
                st.subheader("Feature Importance")
                importance = model.get_score(importance_type='weight')
                importance_df = pd.DataFrame(list(importance.items()), 
                                           columns=['Feature', 'Importance'])
                importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                      orientation='h', title="Top 10 Feature Importance")
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                st.subheader("Sales Trends")
                yearly_sales = sales_data.groupby('Year')['Sales'].sum().reset_index()
                fig_yearly = px.line(yearly_sales, x='Year', y='Sales', 
                                   title="Yearly Sales Trend")
                st.plotly_chart(fig_yearly, use_container_width=True)
                
                st.subheader("Price vs Sales")
                fig_price_sales = px.scatter(sales_data, x='Avg_Price', y='Sales', 
                                           title="Price vs Sales Relationship")
                st.plotly_chart(fig_price_sales, use_container_width=True)
        else:
            st.error("Model not trained. Please check your data.")

else:
    st.info("👆 Please upload your car sales CSV file to get started!")
    
    # Show sample data format
    st.subheader("Expected Data Format")
    st.markdown("""
    Your CSV file should contain columns like:
    - Date
    - Customer Name
    - Gender
    - Annual Income
    - Dealer_Name
    - Company
    - Model
    - Engine
    - Transmission
    - Color
    - Price ($)
    - Body Style
    - Dealer_Region
    - Phone
    - Dealer_No
    - Car_id
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Car Sales Prediction Dashboard")
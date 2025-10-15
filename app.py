import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Configuration ---
st.set_page_config(
    page_title="Product Price Predictor",
    page_icon="📊",
    layout="wide"
)

# --- 1. Load Dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("books_dataset.csv")
        df = df.dropna(subset=['price', 'original_price', 'category', 'brand'])
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'books_dataset.csv' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# --- 2. Load Model and Encoders ---
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load("xgb_product_model.joblib")
        encoders = joblib.load("label_encoders.joblib")
        return model, encoders
    except FileNotFoundError as e:
        st.error(f"❌ Error: Model or encoder file not found. {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load data and model
df = load_data()
model, encoders = load_model_and_encoders()

# --- 3. Streamlit UI ---
st.title("📊 Product Price Predictor")
st.write("Use this app to predict product prices and explore feature importance.")

# Display dataset info
with st.expander("📋 Dataset Information"):
    st.write(f"**Total Products:** {len(df)}")
    st.write(f"**Features:** {', '.join(df.columns.tolist())}")
    st.dataframe(df.head())

# Sidebar for user input
st.sidebar.header("Input Product Details")

def user_input_features():
    original_price = st.sidebar.number_input("Original Price ($)", min_value=0.0, value=100.0, step=1.0)
    rating = st.sidebar.slider("Rating", 0.0, 5.0, 4.0, 0.1)
    rating_count = st.sidebar.number_input("Number of Ratings", min_value=0, value=50, step=1)
    
    # Check if required columns exist
    required_cols = ['category', 'brand', 'availability']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Column '{col}' not found in dataset")
            st.stop()
    
    category = st.sidebar.selectbox("Category", sorted(df['category'].unique()))
    brand = st.sidebar.selectbox("Brand", sorted(df['brand'].unique()))
    
    # Handle availability column with fallback
    if 'availability' in df.columns:
        availability = st.sidebar.selectbox("Availability", sorted(df['availability'].unique()))
    else:
        availability = "In Stock"  # Default value
        st.sidebar.warning("⚠️ Availability column not found, using default")
    
    input_df = pd.DataFrame({
        'original_price': [original_price],
        'rating': [rating],
        'rating_count': [rating_count],
        'category': [category],
        'brand': [brand],
        'availability': [availability]
    })
    
    # Encode categorical features using saved encoders
    encoded_input = input_df.copy()
    for col in ['category', 'brand', 'availability']:
        if col in encoders:
            le = encoders[col]
            try:
                encoded_input[col] = le.transform(input_df[col])
            except ValueError as e:
                st.error(f"❌ Error encoding '{col}': {str(e)}")
                st.stop()
        else:
            st.error(f"❌ Encoder for '{col}' not found")
            st.stop()
    
    return input_df, encoded_input

input_df_raw, input_df_encoded = user_input_features()

# --- 4. Prediction ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Summary")
    st.dataframe(input_df_raw)

with col2:
    st.subheader("Predicted Price")
    try:
        prediction = model.predict(input_df_encoded)[0]
        st.markdown(f"### 💰 ${prediction:.2f}")
        
        # Show comparison with original price
        original = input_df_raw['original_price'].values[0]
        difference = prediction - original
        percentage = (difference / original * 100) if original > 0 else 0
        
        if difference > 0:
            st.success(f"📈 +${difference:.2f} ({percentage:.1f}%) above original price")
        elif difference < 0:
            st.info(f"📉 -${abs(difference):.2f} ({abs(percentage):.1f}%) below original price")
        else:
            st.info("➡️ Same as original price")
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

# --- 5. SHAP Analysis ---
st.divider()
st.subheader("🔍 Feature Importance (SHAP)")

# Add toggle for SHAP analysis
show_shap = st.checkbox("Show SHAP Analysis (may take a moment to compute)", value=False)

if show_shap:
    try:
        with st.spinner("Computing SHAP values..."):
            # Encode the dataset for model
            X_model = df[['original_price', 'rating', 'rating_count', 'category', 'brand', 'availability']].copy()
            
            # Encode categorical features
            for col in ['category', 'brand', 'availability']:
                if col in encoders:
                    X_model[col] = encoders[col].transform(X_model[col])
            
            # Use a sample for faster computation
            sample_size = min(1000, len(X_model))
            X_sample = X_model.sample(n=sample_size, random_state=42)
            
            # Compute SHAP values
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            # Create two columns for plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**SHAP Summary Plot (Bar)**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.write("**SHAP Summary Plot (Beeswarm)**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
                plt.close()
            
            # SHAP for current prediction
            st.write("**SHAP Explanation for Current Prediction**")
            shap_values_single = explainer(input_df_encoded)
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.waterfall_plot(shap_values_single[0], show=False)
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.error(f"❌ SHAP analysis error: {str(e)}")
        st.write("Please ensure SHAP is properly installed and the model is compatible.")

# --- Footer ---
st.divider()
st.caption("📊 Product Price Predictor | Built with Streamlit & XGBoost")
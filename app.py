import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Sales Predictor Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Assets
# ----------------------------
@st.cache_resource
def load_assets():
    try:
        assets = joblib.load("sales_classifier.joblib")
        return assets
    except FileNotFoundError:
        st.error("Asset file 'sales_classifier.joblib' not found. Please ensure it's in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the assets: {e}")
        return None

# Load the models, scaler, and columns
assets = load_assets()

# Check if assets were loaded successfully
if assets is None:
    st.stop()
    
# Extract assets from the loaded dictionary
models = assets.get('models')
scaler = assets.get('scaler')
model_columns = assets.get('columns')
# ----------------------------
# Prediction Logic
# ----------------------------
def predict(input_df, model, scaler, columns):
    """Predicts sales class for a given DataFrame of input data."""
    
    # 1. Preprocess the new data
    processed_df = input_df.copy()
    processed_df['Item_Weight'].fillna(processed_df['Item_Weight'].median(), inplace=True)
    
    cat_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        processed_df[c].fillna(processed_df[c].mode()[0], inplace=True)
        
    # 2. One-hot encode the categorical features
    processed_encoded = pd.get_dummies(processed_df, drop_first=True)
    
    # 3. Align columns with the training data (important for unseen data)
    for col in columns:
        if col not in processed_encoded.columns:
            processed_encoded[col] = 0
            
    processed_encoded = processed_encoded[columns]
    
    # 4. Scale the features
    scaled_features = scaler.transform(processed_encoded)
    
    # 5. Make a prediction
    predictions = model.predict(scaled_features)
    return predictions

# ----------------------------
# UI Layout
# ----------------------------
st.sidebar.title("Model Selection")
st.sidebar.markdown("Choose a model from the list below to perform predictions.")

# Model selection in the sidebar
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox(
    "Select Model",
    model_names,
    index=model_names.index("SVM_linear") if "SVM_linear" in model_names else 0 # Default to SVM_linear if available
)

selected_model = models.get(selected_model_name)

# Highlight the best model
if selected_model_name == "SVM_linear":
    st.sidebar.success("💡 **Your best model!** This model was identified as the top performer in your analysis.")
    
# Main content area
page = st.sidebar.radio("Navigation", ["Home Page", "Single Prediction", "Batch Prediction"])

if page == "Home Page":
    st.title("🛒 Sales Predictor Pro: A Machine Learning Classifier")
    st.markdown("""
        Welcome to the Sales Predictor Pro application. This tool leverages machine learning models to classify sales data 
        into three categories: **Low**, **Medium**, and **High**.
        
        This application was developed as a working prototype based on your extensive data analysis and model training efforts.
        
        ### Key Features:
        - **Data Cleaning and Preprocessing:** Your data is automatically cleaned and prepared before being fed to the models.
        - **Multiple Model Support:** You can select from a variety of trained algorithms, including SVM, KNN, and ANN.
        - **Single & Batch Predictions:** Get instant predictions for a single item or upload a CSV file to get results for many items at once.
        - **Best Model Highlight:** The UI highlights your top-performing model, **`SVM_linear`**, as identified in your analysis.
        
        ### Get Started:
        1. Navigate to either "Single Prediction" or "Batch Prediction" in the sidebar.
        2. Choose your preferred model.
        3. Enter the required data and get your predictions!
    """)
    st.image("https://images.unsplash.com/photo-1579621970795-87facc2f976d?q=80&w=1470&auto=format&fit=crop", 
             caption="Dashboard visualization of sales data", use_column_width=True)

elif page == "Single Prediction":
    st.title("Single Prediction")
    st.markdown(f"**Selected Model:** `{selected_model_name}`")
    
    with st.form("single_prediction_form"):
        st.header("Enter Item Details")
        
        # Two columns for a cleaner layout
        col1, col2 = st.columns(2)
        with col1:
            item_weight = st.number_input("Item_Weight", min_value=0.0, value=12.8, format="%.2f")
            item_fat_content = st.selectbox("Item_Fat_Content", ["Low Fat", "Regular", "low fat", "reg", "LF"])
            item_visibility = st.number_input("Item_Visibility", min_value=0.0, value=0.06, format="%.4f")
            item_type = st.selectbox("Item_Type", ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 
                                                   'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 
                                                   'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 
                                                   'Others', 'Seafood'])
        with col2:
            item_mrp = st.number_input("Item_MRP", min_value=0.0, value=150.0, format="%.2f")
            outlet_establishment_year = st.selectbox("Outlet_Establishment_Year", range(1985, 2010))
            outlet_size = st.selectbox("Outlet_Size", ["Small", "Medium", "High"])
            outlet_location_type = st.selectbox("Outlet_Location_Type", ["Tier 1", "Tier 2", "Tier 3"])
            outlet_type = st.selectbox("Outlet_Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
            
        submitted = st.form_submit_button("Predict Sales Class")
        
        if submitted:
            # Create a DataFrame from the inputs
            single_item_data = {
                'Item_Weight': [item_weight],
                'Item_Fat_Content': [item_fat_content],
                'Item_Visibility': [item_visibility],
                'Item_Type': [item_type],
                'Item_MRP': [item_mrp],
                'Outlet_Establishment_Year': [outlet_establishment_year],
                'Outlet_Size': [outlet_size],
                'Outlet_Location_Type': [outlet_location_type],
                'Outlet_Type': [outlet_type]
            }
            input_df = pd.DataFrame(single_item_data)
            
            # Make the prediction
            prediction = predict(input_df, selected_model, scaler, model_columns)
            
            st.markdown("---")
            st.subheader("Prediction Result")
            st.success(f"The predicted sales class for this item is: **{prediction[0]}**")
            
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown(f"**Selected Model:** `{selected_model_name}`")
    st.markdown("Upload a CSV file with your data to get batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of the uploaded data:")
            st.dataframe(batch_df.head())
            
            if st.button("Run Batch Prediction"):
                required_cols = [
                    'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 
                    'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 
                    'Outlet_Location_Type', 'Outlet_Type'
                ]
                
                if not all(col in batch_df.columns for col in required_cols):
                    st.error(f"Error: The uploaded CSV must contain the following columns: {', '.join(required_cols)}")
                else:
                    predictions = predict(batch_df, selected_model, scaler, model_columns)
                    result_df = batch_df.copy()
                    result_df['Predicted_Sales_Class'] = predictions
                    
                    st.subheader("Batch Prediction Results")
                    st.dataframe(result_df)

                    csv_results = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_results,
                        file_name='prediction_results.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

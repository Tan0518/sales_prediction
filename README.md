
# 🛒 Sales Predictor Pro

## Introduction

### 1. About the Dataset
The system is built on a retail sales dataset containing product attributes such as **Item Weight, Item Visibility, Item MRP (price), Item Type, and Outlet details**. These features are used to classify products into **High, Medium, or Low sales categories**, helping businesses understand product performance.

Demo Streamlit link:https://salesprediction-42qh5fyfwswlyoubnr6zw5.streamlit.app/

### 2. Repository Structure
- **app.py** → Main Streamlit application file that runs the web interface.  
- **requirements.txt** → List of required Python libraries for installation.  
- **sales_classifier.joblib** → Pre-trained machine learning models, scaler, and feature information (loaded during execution).  
- **data/** *(optional)* → Folder for storing input CSV files for batch prediction.  

### 3. Installation and Running
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd <your-repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Access the system in your browser at `http://localhost:8501`.

### 4. How Does the System Work?
The system loads **AI models (SVM, KNN, ANN)** from a saved `.joblib` file. 
Users can:
- View a homepage to know about the features of the modules, comparision of default models and tuned models. 
- Enter product details in the sidebar for **single predictions**, to predit whether it is low, medium or high. 
- Upload a CSV file for **batch predictions**, and can download the result. 

The model processes input features, applies data preprocessing (encoding & scaling), and predicts the **sales category**. The results are displayed in an interactive interface, with options to download predictions as a CSV.

import streamlit as st
import pandas as pd
import model as ml_logic
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Agriculture Crop Production Prediction", layout="wide")

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🌾 Agriculture Crop Production Prediction in India")
st.markdown("---")

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Phase", ["Upload & Preview", "EDA", "Train & Evaluate", "Prediction Form"])

# Helper function to load and cache data
@st.cache_data
def get_data(file):
    return ml_logic.load_data(file)

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    st.session_state.data = get_data(uploaded_file)
    st.sidebar.success("File Uploaded successfully!")

# Phase 1: Upload & Preview
if options == "Upload & Preview":
    st.header("📂 Dataset Preview & Cleaning")
    if st.session_state.data is not None:
        df = st.session_state.data
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Shape: {df.shape}")
        with col2:
            st.info(f"Missing Values: {df.isnull().sum().sum()}")
            
        if st.button("Perform Data Cleaning"):
            df_cleaned = ml_logic.clean_data(df)
            st.session_state.data = df_cleaned
            st.success("Duplicates & Missing Values removed.")
            st.dataframe(df_cleaned.head())
    else:
        st.warning("Please upload a CSV file from the sidebar.")

# Phase 2: EDA
elif options == "EDA":
    st.header("📊 Exploratory Data Analysis")
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("Production Visualizations")
        
        # Select chart type
        chart_type = st.selectbox("Select Chart Type", ["Crop-wise Production", "State-wise Production", "Season-wise Production"])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        if chart_type == "Crop-wise Production":
            df.groupby('Crop')['Production'].sum().plot(kind='bar', ax=ax, color='teal')
            ax.set_title("Total Production per Crop")
        elif chart_type == "State-wise Production":
            df.groupby('State')['Production'].sum().plot(kind='bar', ax=ax, color='orange')
            ax.set_title("Total Production per State")
        else:
            df.groupby('Season')['Production'].sum().plot(kind='bar', ax=ax, color='green')
            ax.set_title("Total Production per Season")
        
        st.pyplot(fig)
    else:
        st.error("No data available for EDA. Please upload a dataset.")

# Phase 3: Train & Evaluate
elif options == "Train & Evaluate":
    st.header("⚙️ Model Training & Evaluation")
    if st.session_state.data is not None:
        df = st.session_state.data
        
        if st.button("Train Models"):
            try:
                with st.spinner("Training models... please wait."):
                    X_train, X_test, y_train, y_test, le_crop, le_state, le_season = ml_logic.preprocess_data(df)
                    
                    if len(X_train) == 0:
                        st.error("Error: Dataset is empty after removing missing values. Please check your data.")
                    else:
                        # Train Models
                        rf_model = ml_logic.train_rf_model(X_train, y_train)
                        lr_model = ml_logic.train_lr_model(X_train, y_train)
                        
                        # Store in session state
                        st.session_state.model = rf_model
                        st.session_state.encoders = (le_crop, le_state, le_season)
                        
                        # Evaluation
                        st.subheader("Model Performance")
                        
                        # Random Forest
                        r2_rf, mse_rf, fig_rf = ml_logic.evaluate_model(rf_model, X_test, y_test)
                        # Linear Regression
                        r2_lr, mse_lr, _ = ml_logic.evaluate_model(lr_model, X_test, y_test)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success("**Random Forest Regressor**")
                            st.write(f"R2 Score: {r2_rf:.4f}")
                            st.write(f"MSE: {mse_rf:.2f}")
                        with col2:
                            st.info("**Linear Regression**")
                            st.write(f"R2 Score: {r2_lr:.4f}")
                            st.write(f"MSE: {mse_lr:.2f}")
                        
                        st.subheader("Actual vs Predicted Plot (Random Forest)")
                        st.pyplot(fig_rf)
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                st.info("Check if all required columns ('Crop', 'State', 'Season', 'Area', 'Cost', 'Production') are present and formatted correctly.")
    else:
        st.error("Please upload data before training.")

# Phase 4: Prediction Form
elif options == "Prediction Form":
    st.header("🔮 Production Prediction")
    if st.session_state.model is not None:
        le_crop, le_state, le_season = st.session_state.encoders
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                crop = st.selectbox("Select Crop", le_crop.classes_)
                state = st.selectbox("Select State", le_state.classes_)
                season = st.selectbox("Select Season", le_season.classes_)
            with col2:
                area = st.number_input("Area (in Hectares)", min_value=1.0, value=1000.0)
                cost = st.number_input("Production Cost", min_value=1.0, value=5000.0)
            
            submit = st.form_submit_button("Predict Production")
            
            if submit:
                # Encode inputs
                crop_enc = le_crop.transform([crop])[0]
                state_enc = le_state.transform([state])[0]
                season_enc = le_season.transform([season])[0]
                
                # Predict
                input_data = pd.DataFrame([[crop_enc, state_enc, season_enc, area, cost]], 
                                          columns=['Crop', 'State', 'Season', 'Area', 'Cost'])
                prediction = st.session_state.model.predict(input_data)[0]
                
                st.success(f"### 🌾 Predicted Production: {prediction:.2f} Units")
    else:
        st.warning("Please train the model first in the 'Train & Evaluate' tab.")

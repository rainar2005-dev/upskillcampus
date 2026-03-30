# Agriculture Crop Production Prediction in India 🌾

A complete Machine Learning project built with **Python**, **Streamlit**, and **Scikit-learn** to predict crop production based on various agricultural factors.

## 🚀 Features
- **Upload CSV Dataset**: Load your crop dataset with ease.
- **Data Preview & Cleaning**: View the first 5 rows and handle missing values/duplicates.
- **Exploratory Data Analysis (EDA)**: Beautiful charts for crop, state, and season-wise production.
- **Machine Learning Models**:
  - **Random Forest Regressor** (Primary Model)
  - **Linear Regression** (Performance Comparison)
- **Model Evaluation**: Metrics such as **R2 Score** and **Mean Squared Error (MSE)**.
- **Production Prediction Form**: Calculate predicted production for user inputs.

## 🛠️ Tech Stack
- **UI**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 📂 Project Structure
- `app.py`: Main Streamlit application file.
- `model.py`: Modular code for data processing and model training.
- `generate_sample_data.py`: A helper script to create sample data for testing.
- `requirements.txt`: Project dependencies.

## ⚙️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data (Optional)
If you don't have a dataset, run the following command to generate `crop_production_data.csv`:
```bash
python generate_sample_data.py
```

### 3. Run the App
```bash
streamlit run app.py
```

## 📊 Dataset Columns Required
The application expects a CSV with the following columns:
- `Crop`: Name of the crop (e.g., Rice, Wheat)
- `State`: Indian State (e.g., Punjab, West Bengal)
- `Season`: Season (e.g., Kharif, Rabi)
- `Area`: Cultivated area in hectares
- `Cost`: Cost of production
- `Production`: Target variable (Total production)

## 📝 License
This project is for educational purposes. Feel free to use and modify!

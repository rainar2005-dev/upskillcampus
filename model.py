import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file):
    """Load the CSV dataset."""
    return pd.read_csv(file)

def clean_data(df):
    """Clean the dataset by removing missing values and duplicates."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def preprocess_data(df):
    """Clean data, encode categorical features, and split data."""
    # Ensure no missing values in features or target
    df = df.dropna(subset=['Crop', 'State', 'Season', 'Area', 'Cost', 'Production'])
    
    le_crop = LabelEncoder()
    le_state = LabelEncoder()
    le_season = LabelEncoder()

    df['Crop'] = le_crop.fit_transform(df['Crop'])
    df['State'] = le_state.fit_transform(df['State'])
    df['Season'] = le_season.fit_transform(df['Season'])

    X = df[['Crop', 'State', 'Season', 'Area', 'Cost']]
    y = df['Production']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le_crop, le_state, le_season

def train_rf_model(X_train, y_train):
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lr_model(X_train, y_train):
    """Train Linear Regression."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics and plot."""
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    ax.set_xlabel('Actual Production')
    ax.set_ylabel('Predicted Production')
    ax.set_title('Actual vs Predicted Production')
    
    return r2, mse, fig

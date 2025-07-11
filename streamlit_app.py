import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")

st.title("📈 Walmart Weekly Sales Prediction")
st.write("Upload your dataset to get started.")

# Upload ZIP or CSV files
uploaded_files = st.file_uploader("Upload Walmart Dataset Files (CSV or ZIP)", accept_multiple_files=True)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # If ZIP, extract it
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall("data")

    # Load CSVs
    try:
        train = pd.read_csv("data/train.csv")
        features = pd.read_csv("data/features.csv")
        stores = pd.read_csv("data/stores.csv")
        st.success("✅ Data loaded successfully!")

        # Preprocessing
        df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
        df = df.merge(stores, on='Store', how='left')
        df['Date'] = pd.to_datetime(df['Date'])

        for col in ['Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 
                    'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']:
            df[col] = df[col].fillna(df[col].median())

        df['Type'] = df['Type'].map({'A': 0, 'B': 1, 'C': 2})
        df['IsHoliday'] = df['IsHoliday'].astype(int)
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

        feature_cols = ['Store', 'Dept', 'Type', 'Size', 'Temperature', 'Fuel_Price', 
                        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 
                        'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday', 'Week', 'Month', 'Year']
        target_col = 'Weekly_Sales'

        # Scaling
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X = df[feature_cols]
        y = df[target_col]
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        # Sequence preparation
        sequence_length = 4
        sequences_X, sequences_y = [], []
        for (store, dept), group in df.groupby(['Store', 'Dept']):
            group = group.sort_values('Date')
            X_group = scaler_X.transform(group[feature_cols])
            y_group = scaler_y.transform(group[target_col].values.reshape(-1, 1))
            for i in range(len(group) - sequence_length):
                sequences_X.append(X_group[i:i+sequence_length])
                sequences_y.append(y_group[i+sequence_length])

        sequences_X = np.array(sequences_X)
        sequences_y = np.array(sequences_y)

        # Model selection
        model_choice = st.selectbox("Choose a model", ["LSTM", "CNN"])
        st.write("Training the model (5 epochs)...")

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(sequences_X, sequences_y, test_size=0.2, random_state=42)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Flatten

        if model_choice == "LSTM":
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
        else:
            model = Sequential([
                Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(32, activation='relu'),
                Dense(1)
            ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=0)

        # Evaluation
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f} | **R²:** {r2:.4f}")

        # Plotting
        st.subheader("📉 Actual vs Predicted Sales")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true[:100], label='Actual')
        ax.plot(y_pred[:100], label='Predicted')
        ax.legend()
        ax.set_title("Actual vs Predicted Weekly Sales")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error loading or processing files: {e}")

import torch
from datetime import datetime
import numpy as np
import joblib
import pandas as pd
from src.market_model import MarketLSTM
import config
import os

# --- Configuration ---
MODEL_PATH = "models/market_model.pth"
SCALER_PATH = f"data/{config.DATA_FILE_NAME}_scaler.save"
SEQUENCE_LENGTH = 50
HIDDEN_LAYER_SIZE = 50

def load_resources():
    """Loads the model and the scaler."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return None, None
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler not found at {SCALER_PATH}. Please run processing script first.")
        return None, None

    # Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load Model
    # We need to know the input size. Since it's BTC data (Open, High, Low, Close, Volume), it's 5.
    # In a more robust system, we could save this metadata.
    input_size = 5 
    model = MarketLSTM(input_size=input_size, hidden_layer_size=HIDDEN_LAYER_SIZE, output_size=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    return model, scaler

def predict_next_price(model, scaler, recent_data):
    """
    Predicts the next price based on a sequence of recent data.
    recent_data: pandas DataFrame or numpy array of shape (SEQUENCE_LENGTH, 5)
    """
    # 1. Scale the data
    scaled_data = scaler.transform(recent_data)
    
    # 2. Convert to tensor and add batch dimension (1, SEQ_LEN, FEATURES)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    
    # 3. Predict
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    # 4. Inverse transform to get actual price
    # The scaler was fit on 5 columns. We need to create a dummy array to inverse transform.
    # Target was column 3 (Close price)
    prediction_val = prediction_scaled.item()
    
    # Create a dummy row for inverse_transform
    dummy = np.zeros((1, 5))
    dummy[0, 3] = prediction_val # Put prediction in the 'Close' column index
    
    # Inverse transform and extract the 'Close' price
    unscaled_prediction = scaler.inverse_transform(dummy)[0, 3]
    
    return unscaled_prediction

def main():
    print("--- Market Price Predictor ---")
    
    model, scaler = load_resources()
    if model is None:
        return

    # Load the original data to get the latest sequence for a demo prediction
    print(f"Loading data from {config.DATA_PATH} for demo prediction...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Prepare data (drop Timestamp like we do in DataProcessor)
    data_only = df.drop(columns=['Timestamp'])
    
    # Take the last 50 samples
    last_sequence = data_only.tail(SEQUENCE_LENGTH)
    
    if len(last_sequence) < SEQUENCE_LENGTH:
        print("Error: Not enough data for a prediction.")
        return

    current_price = last_sequence.iloc[-1, 3] # Last 'Close' price
    predicted_price = predict_next_price(model, scaler, last_sequence)
    
    print("Last data timestamp:", datetime.fromtimestamp(df['Timestamp'].iloc[-1]).strftime('%d/%m/%Y, %H:%M:%S'))
    print("\n" + "="*30)
    print(f"Current Price (Close): {current_price:.2f}")
    print(f"Predicted Price (Next Min): {predicted_price:.2f}")
    
    difference = predicted_price - current_price
    direction = "UP" if difference > 0 else "DOWN"
    print(f"Signal: {direction} ({difference:+.2f})")
    print("="*30)

if __name__ == "__main__":
    main()

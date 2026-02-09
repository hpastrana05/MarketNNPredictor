import sys
import os

# Add the project root to the python path so we can import from src and config
# This allows the script to be run from anywhere, e.g., 'python scripts/model_training.py' or 'python -m scripts.model_training'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from src.market_model import MarketLSTM
import config

# --- Hyperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_LAYER_SIZE = 50

def load_preprocessed_data():
    """Loads the preprocessed numpy data saved by the processing script."""
    # Construct the path based on config, consistent with processing_data.py
    data_path = f"data/{config.DATA_FILE_NAME}_preprocessed.npz"
    print(f"Loading preprocessed data from {data_path}...")
    
    try:
        # Load the data using joblib as it was saved with joblib in processing_data.py
        data = joblib.load(data_path)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        print("Please run 'scripts/processing_data.py' first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    print("--- Market Model Training Script ---")
    
    # 1. Load Data
    data = load_preprocessed_data()
    if data is None:
        return

    # Unpack and immediately delete the original tuple to free memory
    X_train, y_train, X_test, y_test = data
    del data
    
    print(f"Data loaded successfully.")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input features: {X_train.shape[2]}")

    # Ensure data is float32 if it wasn't already (backwards compatibility for old saved data)
    import numpy as np
    if X_train.dtype != np.float32:
        print("Converting training data to float32...")
        X_train = X_train.astype(np.float32)
    if X_test.dtype != np.float32:
        print("Converting test data to float32...")
        X_test = X_test.astype(np.float32)
    if y_train.dtype != np.float32:
        y_train = y_train.astype(np.float32)
    if y_test.dtype != np.float32:
        y_test = y_test.astype(np.float32)

    # 2. Initialize Model
    # input_size is the number of features (columns) in the data (last dimension)
    input_size = X_train.shape[2] 
    
    model = MarketLSTM(
        input_size=input_size, 
        hidden_layer_size=HIDDEN_LAYER_SIZE, 
        output_size=1
    )

    # 3. Train Model
    # We use the train_model method we added to the class
    save_path = "models/market_model.pth"
    
    model.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_path=save_path
    )
    
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
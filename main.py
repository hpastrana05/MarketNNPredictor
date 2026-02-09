import joblib
from src.market_model import MarketLSTM
import config

# --- Hyperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_LAYER_SIZE = 50
SAVE_MODEL_PATH = "models/market_model.pth"

def load_preprocessed_data():
    """Loads the preprocessed numpy data saved by the script."""
    data_path = f"data/{config.DATA_FILE_NAME}_preprocessed.npz"
    print(f"Loading preprocessed data from {data_path}...")
    try:
        data = joblib.load(data_path)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main():
    # 1. Load Data
    data = load_preprocessed_data()
    if data is None:
        return
        
    X_train, y_train, X_test, y_test = data

    # 2. Initialize Model
    # input_size is the number of features (columns) in the data
    input_size = X_train.shape[2] 
    model = MarketLSTM(
        input_size=input_size, 
        hidden_layer_size=HIDDEN_LAYER_SIZE, 
        output_size=1
    )

    # 3. Train Model
    # The training logic is now encapsulated within the model class
    model.train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_path="market_model.pth"
    )

if __name__ == "__main__":
    main()
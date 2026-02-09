from src.data_processor import DataProcessor
from config import DATA_PATH, DATA_FILE_NAME

manager = DataProcessor(data_path=DATA_PATH, target_column=3, sequence_length=50)

data = manager.data
print(data.head())

X_train, y_train, X_test, y_test = manager.preprocess_data(train=0.8)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

manager.save_scaler(f'{DATA_FILE_NAME}_scaler.save')
manager.save_preprocessed_data(f'{DATA_FILE_NAME}_preprocessed.npz')


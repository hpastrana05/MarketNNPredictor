# MarketNNPredictor

MarketNNPredictor is a project designed for time-series forecasting, specifically utilizing a PyTorch Long Short-Term Memory (LSTM) neural network to predict market trends. The project follows a structured workflow for data acquisition, preprocessing, model training, and prediction.

## Future improvements
- Data should be compressed other way (RobustScaler, StandardScaler, Scale windows locally)
- Data preprocessing shouldn't split the data for train and test
- Model training should properly set the data for training and testing
- Make a recursive prediction for having more data predicted
- Check if new predictions are right
- Print the predicted candles plot to see what is predicted and then print real ones for seeing if it "fits"


## Project Structure

The project is organized into several key directories and files:

-   `.gitignore`: Specifies intentionally untracked files to ignore.
-   `config.py`: Centralized configuration settings for the project.
-   `main.py`: The main entry point for making predictions using a trained model.
-   `README.md`: This file, providing an overview of the project.
-   `data/`: Stores processed data and data scalers after preprocessing.
-   `scripts/`: Contains executable scripts for various stages of the workflow.
-   `src/`: Holds the core source code, including data processing and model definitions.
-   `upload/`: A temporary directory for raw input data before processing.
-   `.venv/`: Virtual environment for managing project dependencies.
-   `__pycache__/`: Python's bytecode cache directories.
-   `.git/`: Git version control system directory.

## Core Components

### `main.py`
This script serves as the primary entry point for using the trained market prediction model. It loads a pre-trained model and a data scaler, then uses them to predict the next market price based on the latest available data.

### `config.py`
This file manages project-wide configurations, such as the name and location of the input data file. It is crucial for defining where the application expects to find its raw data.

### `src/data_processor.py`
This module contains the `DataProcessor` class, which handles all aspects of data preparation. Its responsibilities include:
-   Loading raw market data.
-   Splitting data into training and testing sets.
-   Scaling numerical features using `MinMaxScaler`.
-   Transforming sequential data into suitable input format for the LSTM model.
-   Saving the fitted scaler for later use during prediction.

### `src/market_model.py`
This module defines the neural network architecture used for market prediction. It includes:
-   The `MarketLSTM` class, which implements the PyTorch LSTM model.
-   The `train_model` method, a self-contained training loop used to train the `MarketLSTM` model.

## Scripts

The `scripts/` directory contains automation scripts for the project workflow:

### `scripts/processing_data.py`
This script orchestrates the data preprocessing stage. It utilizes the `DataProcessor` from `src/data_processor.py` to clean, scale, and sequence the raw data found in the `upload/` directory. The processed data and the fitted scaler are then saved into the `data/` directory.

### `scripts/model_training.py`
Following data preprocessing, this script handles the model training phase. It loads the preprocessed data from the `data/` directory, initializes the `MarketLSTM` model from `src/market_model.py`, and executes the training process. The trained model is then saved to a designated `models/` directory (which will be created if it doesn't exist).

### `scripts/load_data/kaggle_update_bitcoin.py`
This script is responsible for acquiring the raw market data. Based on its name, it likely automates the download or update of a Bitcoin dataset from Kaggle, placing it in the `upload/` directory for subsequent processing.

### `scripts/load_data/see_if_worked.py`
This script is likely used to verify if the data acquisition or processing steps were successful, possibly by checking the presence or integrity of the downloaded/processed data files.

### `scripts/test_gpu_working.py` and `scripts/test_gpu.py`
These scripts are utilities for verifying the correct setup and functionality of the GPU environment, ensuring that PyTorch can utilize the available GPU for computation, which is crucial for efficient model training.

## Workflow

The typical workflow for using MarketNNPredictor is as follows:

1.  **Acquire Data**: Run `scripts/load_data/kaggle_update_bitcoin.py` to download the latest market data into the `upload/` directory.
2.  **Process Data**: Execute `scripts/processing_data.py` to clean, scale, and prepare the data for model training. The outputs are saved in the `data/` directory.
3.  **Train Model**: Run `scripts/model_training.py` to train the LSTM model using the processed data. The trained model will be saved for future use.
4.  **Make Predictions**: Use `main.py` to load the trained model and make predictions on new data.
5.  **Verify GPU (Optional)**: If needed, use `scripts/test_gpu_working.py` or `scripts/test_gpu.py` to ensure GPU is properly configured.
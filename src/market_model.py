import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MarketLSTM(nn.Module):

    def __init__(self, input_size=5, hidden_layer_size=50, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # The LSTM layer: processes the sequence
        self.lstm = nn.LSTM(input_size, 
                            hidden_layer_size, 
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout) # Dropout should be between  0.05 and 0.2

        # The Linear layer: maps LSTM output to a single predicted price
        self.linear = nn.Linear(hidden_layer_size, output_size)


    def forward(self, input_seq):
        # Input_seq shape: (batch_size, sequence_length, input_size)
        out, (hn, cn) = self.lstm(input_seq)

        # Takes the output of hte LAST time step in the sequence
        pred = self.linear(out[:, -1, :])
        return pred

    def train_model(self, X_train, y_train, X_test=None, y_test=None, epochs=20, batch_size=64, learning_rate=0.001, save_path="market_model.pth"):
        """
        Trains the LSTM model.
        """
        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"Training on device: {device}")

        # Convert to tensors (keep on CPU for now to save GPU memory)
        # Using as_tensor or from_numpy can be more memory efficient
        X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_loader = None
        if X_test is not None and y_test is not None:
             X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32)
             y_test_tensor = torch.as_tensor(y_test, dtype=torch.float32).view(-1, 1)
             test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
             test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        print("Starting training...")
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            
            # Add progress bar for training batches
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch")
            for batch_X, batch_y in pbar:
                # Move batch to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                running_loss += loss_val
                pbar.set_postfix({"loss": f"{loss_val:.6f}"})

            avg_train_loss = running_loss / len(train_loader)
            
            log_msg = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}"

            if test_loader:
                self.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        # Move batch to device
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        
                        predictions = self(batch_X)
                        loss = criterion(predictions, batch_y)
                        test_loss += loss.item()
                avg_test_loss = test_loss / len(test_loader)
                log_msg += f", Test Loss: {avg_test_loss:.6f}"
            
            print(log_msg)

        if save_path:
            torch.save(self.state_dict(), save_path)
            print(f"Model saved to {save_path}")
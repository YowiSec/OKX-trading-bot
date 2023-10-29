import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load the dataset
data = pd.read_csv('data/data.csv')

# Display the first few rows of the dataset
data.head()

# Extract close prices
close_prices = data['close'].values.reshape(-1, 1)
# Calculate the log returns for the original data
# data['log_return'] = np.log(data['close'] / data['close'].shift(1))
# Drop the NaN value created by the log return calculation
# data = data.dropna()
# Normalize the close prices
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(close_prices)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences
SEQ_LENGTH = 10
X, y = create_sequences(normalized_data, SEQ_LENGTH)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

X_train.shape, y_train.shape, X_test.shape, y_test.shape
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(DQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x, hidden):
        lstm_out, _ = self.lstm(x, hidden)
        output = self.linear(lstm_out[:, -1, :])
        return output
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode

# Define the model
input_dim = 1  # since we are only using 'close' prices
hidden_dim = 50

model = DQN(input_dim, hidden_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Training parameters
num_epochs = 100
batch_size = 64

# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Shuffle data for mini-batch gradient descent
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_tensor = X_train_tensor[indices]
    y_train_tensor = y_train_tensor[indices]
    
    for i in range(0, len(X_train), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]

        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0))
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs, hidden)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
    # Track training loss
    train_losses.append(loss.item())
    
    # Track validation loss
    with torch.no_grad():
        h_test, c_test = model.init_hidden(X_test_tensor.size(0))
        test_outputs = model(X_test_tensor, (h_test, c_test))
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

#train_losses[-1], test_losses[-1]
# Create a figure with two subplots
fig = go.Figure()

# Add training loss trace
fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))

# Add testing loss trace
fig.add_trace(go.Scatter(y=test_losses, mode='lines', name='Test Loss'))

# Update layout for the first subplot
fig.update_layout(title='Training & Testing Loss over Epochs',
                  xaxis_title='Epochs',
                  yaxis_title='Loss')

# Display the figure
#fig.show()

# Specify the path where you want to save the model weights
save_path = "trained_model.pth"

# Save the model weights
model.save_model(save_path)


model = DQN(input_dim, hidden_dim).to(device)
model.load_model("trained_model.pth")

# Assuming 'new_data' is your new data point or batch of data points
new_data = pd.read_csv('data/newdata.csv')
# Calculate the log returns for the new_data
# new_data['log_return'] = np.log(new_data['close'] / new_data['close'].shift(1))
# new_data = new_data.dropna()
# Extract close prices
# close_prices2 = new_data['log_return'].values.reshape(-1, 1)
close_prices2 = new_data['close'].values.reshape(-1, 1)

# Normalize the close prices
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(close_prices2)

# Normalize the new data using the same scaler you used for training
normalized_new_data = scaler.transform(close_prices2)


# Create sequences for the new data
X, y  = create_sequences(normalized_new_data, SEQ_LENGTH)
#X, y = create_sequences(normalized_data, SEQ_LENGTH)
#print(new_sequences)
#print(new_sequences.shape)
# Convert to PyTorch tensor and move to device
new_sequences_tensor = torch.FloatTensor(X).to(device)

# Predict using the model
with torch.no_grad():
    h_new, c_new = model.init_hidden(new_sequences_tensor.size(0))
    predictions = model(new_sequences_tensor, (h_new, c_new))

threshold = 0.01  # for example, 1% price difference to trigger a trade

# Generating trading signals
signals = []
for pred, actual in zip(predictions, normalized_new_data[:, -1]):
    pred = float(pred)
    #print("pred: %s" % pred)
    #print("actual: %s" % actual)
    if pred - actual > threshold:
        signals.append('Buy')
    elif pred - actual < -threshold:
        signals.append('Sell')
    else:
        signals.append('Hold')


initial_balance = 10000  # starting with $10,000 for example
balance = initial_balance
stock_quantity = 0
# Use raw prices for transactions
prices = new_data['close'].values[-len(signals):]

# Use log returns for buy/sell decisions
# returns = new_data['log_return'].values[-len(signals):]

allocation = 0.1  # Or whatever percentage you decide

for i, signal in enumerate(signals):
    print("CURRENT SIGNAL: %s" % signal)
    print("BALANCE: %s" % balance)
    if signal == 'Buy':
        buy_amount = allocation * balance
        if buy_amount >= prices[i]:
            buy_quantity = buy_amount // prices[i]
            balance -= buy_quantity * prices[i]
            stock_quantity += buy_quantity
    elif signal == 'Sell' and stock_quantity > 0:
        balance += stock_quantity * prices[i]
        stock_quantity = 0


# Final value
final_balance = balance + stock_quantity * (prices[-1] if prices[-1] else 0)

profit_or_loss = final_balance - initial_balance
print("P/L: %s" % profit_or_loss)
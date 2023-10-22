import pandas as pd

# Load the dataset
data = pd.read_csv('data.csv')

# Display the first few rows of the dataset
data.head()

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.optim as optim
import random
import torch
import torch.nn as nn

# Extract the 'close' prices
close_prices = data['close'].values.reshape(-1, 1)

# Normalize the 'close' prices
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_prices)

# Create sequences of data for training the LSTM
sequence_length = 10  # Number of historical data points to use for prediction

X = []
y = []

for i in range(sequence_length, len(scaled_close)):
    X.append(scaled_close[i-sequence_length:i])
    y.append(scaled_close[i])

X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train.shape, y_train.shape, X_test.shape, y_test.shape

class TradingEnvironment:
    def __init__(self, data, initial_balance=1000):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None  # Store the price at which we bought
        self.current_step = None
        self.done = None

    def reset(self):
        self.balance = self.initial_balance
        self.position = None
        self.current_step = 0
        self.done = False
        return self.data[self.current_step]

    def step(self, action):
        reward = 0

        # If we're at the end of the data, end the episode
        if self.current_step == len(self.data) - 1:
            self.done = True

        else:
            self.current_step += 1

            # If we have a position and decide to sell
            if action == 1 and self.position is not None:
                reward = self.data[self.current_step] - self.position  # Profit or loss
                self.position = None

            # If we don't have a position and decide to buy
            elif action == 0 and self.position is None:
                self.position = self.data[self.current_step]

        # Calculate the current portfolio value
        if self.position is not None:
            current_portfolio = self.balance + (self.data[self.current_step] - self.position)
        else:
            current_portfolio = self.balance

        # Check if we've gone bankrupt
        if current_portfolio <= 0:
            self.done = True

        return self.data[self.current_step], reward, self.done

    def action_space(self):
        return [0, 1, 2]  # Buy, Sell, Hold

    def observation_space(self):
        return self.data[self.current_step]


# Initialize the trading environment
env = TradingEnvironment(data=scaled_close)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last LSTM output
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the DQN model
input_dim = 1  # Only 'close' prices
output_dim = 3  # Buy, Sell, Hold actions
model = DQN(input_dim, output_dim)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 0.3  # Exploration rate
num_epochs = 10
batch_size = 32

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    state = env.reset()
    episode_loss = 0
    
    for t in range(len(env.data) - 1):
        # Select action with epsilon-greedy strategy
        if random.random() < epsilon:
            action = random.choice(env.action_space())
        else:
            q_values = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()

        # Take action in the environment
        next_state, reward, done = env.step(action)

        # Compute target Q-value
        with torch.no_grad():
            next_q_values = model(torch.FloatTensor(next_state).unsqueeze(0))
            target_q_value = reward + gamma * torch.max(next_q_values)

        # Compute expected Q-value
        q_values = model(torch.FloatTensor(state).unsqueeze(0))
        expected_q_value = q_values[0][action]

        # Compute loss and update model
        loss = criterion(expected_q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_loss += loss.item()
        state = next_state

        if done:
            break

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {episode_loss:.4f}")

model.eval()  # Set the model to evaluation mode

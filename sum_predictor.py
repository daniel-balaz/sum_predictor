import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL
class SumPredikce(nn.Module):
    def __init__(self):
        super(SumPredikce, self).__init__()
        self.hidden = nn.Linear(2, 10)
        self.output = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# MODEL INICIALIZATION
model = SumPredikce().to(device)

# LOAD SAVED MODEL, IF EXIST
if os.path.exists("sum_predictor.pth"):
    model.load_state_dict(torch.load("sum_predictor.pth"))
    model.eval()
    print("Model načten!")
else:
    x_train = torch.randint(low=0, high=100, size=(1000, 2), dtype=torch.float32).to(device)
    y_train = torch.sum(x_train, dim=1, keepdim=True).float().to(device)

    criteria = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3000

    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(x_train)
        loss = criteria(prediction, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: [{epoch}/{epochs}]\nLoss: {loss.item():.2f}")
            
        torch.save(model.state_dict(), "sum_predictor.pth")
        print("Model uložen jako sum_predictor.pth")

# TESTING DATA
x_test = torch.tensor([[20, 30], [50, 50], [10, 5]], dtype=torch.float32).to(device)
y_test = model(x_test).detach().cpu().numpy()

# OUTPUT
print("Prediktované součty:")
for i, x in enumerate(x_test.cpu().numpy()):
    print(f"{x[0]} + {x[1]} = {y_test[i][0]:.2f}")

while True:
    odpoved = input("Chceš znovu trénovat model? (Ano/Ne) ").strip().lower()

    if odpoved == "ano":
        print("Trénujeme nový model...")
        
        model = SumPredikce().to(device)

        if os.path.exists("sum_predictor.pth"):
            model.load_state_dict(torch.load("sum_predictor.pth"))
            print("Uložený model načten.")

        x_train = torch.randint(low=0, high=100, size=(1000, 2), dtype=torch.float32).to(device)
        y_train = torch.sum(x_train, dim=1, keepdim=True).float().to(device)

        criteria = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        epochs = 5000

        for epoch in range(epochs):
            optimizer.zero_grad()
            prediction = model(x_train)
            loss = criteria(prediction, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch: [{epoch}/{epochs}]\nLoss: {loss.item():.2f}")
            
        torch.save(model.state_dict(), "sum_predictor.pth")
        print("Model uložen jako sum_predictor.pth")
    
    else:
        x_test = torch.randint(low=1, high=100, size=(10, 2), dtype=torch.float32).to(device)
        y_test = model(x_test).detach().cpu().numpy()

        print("Prediktované součty:")
        for i, x in enumerate(x_test.cpu().numpy()):
            print(f"{x[0]} + {x[1]} = {y_test[i][0]:.2f}")
        
        torch.save(model.state_dict(), "sum_predictor.pth")
        print("Model uložen jako sum_predictor.pth")
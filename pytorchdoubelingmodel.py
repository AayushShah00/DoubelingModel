import torch
from torch.utils.data import TensorDataset, DataLoader, random_split 
import torch.nn as nn
import torch.optim as optim
import os
X = torch.arange(1, 1001, dtype=torch.float32).unsqueeze(1)
Y = X * 2.0
best_test_loss = float('inf')

full_dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class DoubelingModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1=nn.Linear(1,1)
  def forward(self,x):
    x=self.layer1(x)
    return x


model=DoubelingModel()
loss_fn=torch.nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
epochs = 10000
for epoch in range(epochs):
  model.train()
  for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)
    loss.backward()
    optimizer.step()
    
  if (epoch + 1) % 10 == 0:
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for test_x, test_y in test_loader:
        test_pred = model(test_x)
        test_loss += loss_fn(test_pred, test_y).item()
        
    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch + 1:3d} | Train Loss: {loss.item():10.4f} | Avg Test Loss: {avg_test_loss:10.4f}")
    
    if avg_test_loss < best_test_loss:
      best_test_loss = avg_test_loss
      torch.save(model.state_dict(), 'best_model.pth')
      print(f"--> New best model saved with Test Loss: {best_test_loss:.4f}")




  


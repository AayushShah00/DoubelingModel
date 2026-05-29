import torch
import torch.nn as nn

# 1. Re-create the exact same model structure
class DoubelingModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = nn.Linear(1, 1)
  def forward(self, x):
    x = self.layer1(x)
    return x

# 2. Instantiate the model and load your best weights
best_model = DoubelingModel()
best_model.load_state_dict(torch.load('best_model.pth'))

# 3. Crucial: Put the model in evaluation mode
best_model.eval()

print("Model loaded successfully! Ready for predictions.")
print("------------------------------------------------")

try:
    # 4. Get input from the user
    user_input = input("Enter a number to double: ")
    
    # 5. Convert the string input to a float
    numeric_value = float(user_input)
    
    # 6. Convert to a 2D PyTorch tensor with shape [1, 1] (batch_size=1, features=1)
    tensor_input = torch.tensor([[numeric_value]], dtype=torch.float32)
    
    # 7. Run the model without tracking gradients (saves memory/time)
    with torch.no_grad():
        prediction = best_model(tensor_input)
    
    # 8. Extract the raw number out of the output tensor using .item()
    print(f"\nInput: {numeric_value}")
    print(f"Model Prediction: {prediction.item():.4f}")

except ValueError:
    print("\nError: Please enter a valid numerical value.")

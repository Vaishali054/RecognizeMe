import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from models.lstm import LSTMModel
from tqdm.auto import tqdm
from helper_functions import accuracy_fn, eval_model, train_step, test_step

# Define transforms
data_transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load datasets
train_dataset = datasets.ImageFolder(root="data/FER-2013/train", transform=data_transform)
test_dataset = datasets.ImageFolder(root="data/FER-2013/test", transform=data_transform)

total_dataset_length=len(train_dataset)+len(test_dataset)

# Split train_dataset into train and validation sets
train_size = int(0.7 * total_dataset_length)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the LSTM model
model_0=LSTMModel(input_size=48*48, hidden_size=128, num_layers=2, num_classes=len(emotions)).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_0.parameters(), lr=0.001)

torch.manual_seed(42)
epochs=22

for epoch in tqdm(range(epochs)):
  print(f"Epoch {epoch} \n --------")
  train_step(model= model_0,
             data_loader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             accuracy_fn=accuracy_fn,
             device=device)
  test_step(model= model_0,
             data_loader=test_dataloader,
             loss_fn=loss_fn,
             accuracy_fn=accuracy_fn,
             device=device)


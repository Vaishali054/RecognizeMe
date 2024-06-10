import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from models.lstm import FER2013CNN
from tqdm.auto import tqdm
from helper_functions import accuracy_fn, eval_model, train_step, test_step

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name())

# Define emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define transforms
data_transform = transforms.Compose([
    # transforms.Resize((48, 48)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(48, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

loss_fn = nn.CrossEntropyLoss()


# Load datasets
train_dataset = datasets.ImageFolder(root="data/FER-2013/train", transform=data_transform)
test_dataset = datasets.ImageFolder(root="data/FER-2013/test", transform=data_transform)
total_dataset_length=len(train_dataset)+len(test_dataset)

# Split train_dataset into train and validation sets
test_size = int(0.1 * total_dataset_length)
val_size = len(test_dataset) - test_size
test_dataset, val_dataset = random_split(test_dataset, [test_size, val_size])

# print(f"Training dataset ratio: {len(train_dataset)/total_dataset_length}")
# print(f"Validation dataset ratio: {len(val_dataset)/total_dataset_length}")
# print(f"Test dataset ratio: {len(test_dataset)/total_dataset_length}")

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Instantiate the LSTM model for FER-2013 dataset
# model_0=FER2013CNN(hidden_size=128, num_layers=2, num_classes=len(emotions)).to(device)
model_0=FER2013CNN().to(device)

#Define loss function and optimizer
optimizer = optim.SGD(model_0.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# Train the model on FER-2013 dataset
train_step(model= model_0,
             data_loader=train_dataloader,
            val_loader=val_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             scheduler=scheduler,
             accuracy_fn=accuracy_fn,
             epochs=300,
             checkpoint_path="checkpoint.pth",
             device="cuda")
  
test_step(data_loader=test_dataloader,
          loss_fn=loss_fn,
          accuracy_fn=accuracy_fn,
          device="cuda")

# Saving the model's performance
model_0_results=eval_model(model=model_0,
                           data_loader=test_dataloader,
                           loss_fn=loss_fn,
                           accuracy_fn=accuracy_fn)



# print(f"Training model on dataset RAF-DB\n")

# # Load datasets
# train_dataset_raf = datasets.ImageFolder(root="data/RAF-DB/DATASET/train", transform=data_transform)
# test_dataset_raf = datasets.ImageFolder(root="data/RAF-DB/DATASET/test", transform=data_transform)

# # print(f"Training dataset length: {len(train_dataset_raf)}")
# # print(f"Test dataset length: {len(test_dataset_raf)}")
# total_dataset_length_raf=len(train_dataset_raf)+len(test_dataset_raf)
# # print(f"Total dataset length: {total_dataset_length_raf}")

# # Split train_dataset into train and validation sets
# test_size = int(0.1 * total_dataset_length_raf)
# val_size = len(test_dataset_raf) - test_size
# test_dataset_raf, val_dataset_raf = random_split(test_dataset_raf, [test_size, val_size])

# # print(f"Training dataset ratio: {len(train_dataset_raf)/total_dataset_length_raf}")
# # print(f"Validation dataset ratio: {len(val_dataset_raf)/total_dataset_length_raf}")
# # print(f"Test dataset ratio: {len(test_dataset_raf)/total_dataset_length_raf}")  

# # Create DataLoaders
# train_dataloader_raf = DataLoader(train_dataset_raf, batch_size=128, shuffle=True)
# val_dataloader_raf = DataLoader(val_dataset_raf, batch_size=128, shuffle=False)
# test_dataloader_raf = DataLoader(test_dataset_raf, batch_size=128, shuffle=False)

# model = FER2013CNN().to(device)
# model.load_state_dict(torch.load("best_model.pth"))
# print("Pretrained model on FER-2013 loaded")
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)


# # train_step(model= model,
# #             data_loader=train_dataloader_raf,
# #             val_loader=val_dataloader_raf,
# #             loss_fn=loss_fn,
# #             optimizer=optimizer,
# #             scheduler=scheduler,
# #             accuracy_fn=accuracy_fn,
# #             checkpoint_path="checkpoint_raf.pth",
# #             epochs=300,
# #             device="cuda")

# print(f"\nAfter Training the model on raf-db testing its accuracy on fer-2013")

# test_step(data_loader=test_dataloader,
        #   loss_fn=loss_fn,
        #   accuracy_fn=accuracy_fn,
        #   device="cuda")








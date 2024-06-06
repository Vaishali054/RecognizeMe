import torch
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy_fn(y_true,y_pred):
  correct=torch.eq(y_true,y_pred).sum().item()
  acc=(correct/len(y_pred))*100
  return acc

def eval_model( model : torch.nn.Module,
              data_loader:torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
              device : torch.device=device):
  """Returns a dictionary containing the results of model predicting on data_loader."""
  loss,acc=0,0
  model.eval()
  with torch.inference_mode():
    for X,y in tqdm(data_loader):
      X,y=X.to(device), y.to(device)
      #Make predictions
      y_pred=model(X)

      #Accumulate the loss and acc values per batch
      loss+=loss_fn(y_pred,y)
      acc+=accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    #Scale loss and acc to find the avg loss/acc per batch
    loss/=len(data_loader)
    acc/=len(data_loader)

  return {
      "model_name": model.__class__.__name__, #only works when model was created with class
      "model_loss": loss.item(),
      "model_acc": acc}

# Write a function for train_step

def train_step(model : torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device=device):
  """Performs a training with model trying to learn on data_loader."""

  train_loss, train_acc=0,0

  model.train()
  #Add a loop to loop through the training batches
  for batch, (X,y) in enumerate(data_loader):
    #Put data on target device
    X,y=X.to(device), y.to(device)

    #1. Forward pass
    y_pred=model(X)

    #2 Calculate loss and accuracy (per batch)
    loss=loss_fn(y_pred,y)
    train_loss+=loss
    train_acc+=accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

  #Divice total train loss and acc by length of data loader
  train_loss/=len(data_loader)
  train_acc/=len(data_loader)


  print(f"\nTrain loss : {train_loss:.5f} |  Train acc: {train_acc:.2f}%")

# Write a function for test step

def test_step(model : torch.nn.Module,
              data_loader : torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):

  test_loss, test_acc=0,0

  model.eval()

  with torch.inference_mode():
    for batch,(X,y) in enumerate(data_loader):
      X,y=X.to(device), y.to(device)

      # 1. Forward Pass
      y_test_pred=model(X)

      # 2. Loss and accuracy
      test_loss+=loss_fn(y_test_pred,y)
      test_acc+=accuracy_fn(y_true=y, y_pred=y_test_pred.argmax(dim=1))

    #Calculating average
    test_loss/=len(data_loader)
    test_acc/=len(data_loader)

  print(f"\nTest loss : {test_loss:.5f} |  Test acc: {test_acc:.2f}%")
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import random
from data_preprocessing import image_normalization
from plotting_utils import plot_train_acc
from sklearn.metrics import classification_report
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
    


class Disease_diagnosis(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            #Layer 1 input : 3 * 120 * 180 
            nn.Conv2d(3, 12, kernel_size = 9, stride = 3, padding = 3),
            nn.ReLU(),
            # layer 1 out : 12 * 40 * 60
            nn.Conv2d(12,24, kernel_size = (4, 9), stride = (2, 3), padding = (1, 3)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # Layer 2 out : 24 * 20 * 20
            nn.Conv2d(24, 48, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # Layer 3 out : 48 * 10 * 10
            
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(4800,128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16,3)
        )
    
    def forward(self, x):
        return self.network(x)
    

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train(model, device, train_loader, optimizer, loss_fn, epochs):
    # Set the model to training mode
    model.train()
    
    # Process the images in batches
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = loss_fn(output, target.to(torch.float32))

        # Backpropagate
        loss.backward()
        optimizer.step()
                        
    return 

def evaluation(model, loss_fn, X_train, y_train, X_test, y_test, epoch, class_report=False):
    # model.eval()
    with torch.inference_mode():
        y_logits_train = model(X_train)
        y_pred_train = torch.softmax(y_logits_train, dim=1)
        train_loss = loss_fn(y_logits_train, y_train.to(torch.float32))
        train_acc = accuracy_fn(y_true=y_train.argmax(dim=1), y_pred = y_pred_train.argmax(dim=1))

    
        y_logits_test = model(X_test)
        y_pred_test = torch.softmax(y_logits_test, dim=1)
        test_loss = loss_fn(y_logits_test, y_test.to(torch.float32))
        test_acc = accuracy_fn(y_true=y_test.argmax(dim=1), y_pred=y_pred_test.argmax(dim=1))

    print("Epoch:", epoch, 'Train: Loss: {:.6f}'.format(train_loss), 'Acc: {:.6f}'.format(train_acc), 
            '| Test Loss: {:.6f}'.format(test_loss), 'Test Acc: {:.6f}'.format(test_acc))


    if class_report:
        print(classification_report(torch.Tensor.tolist(y_test.argmax(dim=1)),
                                     torch.Tensor.tolist(y_pred_test.argmax(dim=1)), 
                                     target_names=["acne", "Herpes", "Lichen"]))
    return train_loss, test_loss, train_acc, test_acc



def main():
    st = time.time()
    random.seed(1111)
    np.random.seed(1111)

    # For train and test on base data run these two lines:
    X_train = torch.load("./data/X_train_base.pt")
    y_train = torch.load("./data/y_train_base.pt")


    # For train on augmented data run these two lines:
    # X_train = torch.load("./data/X_train_augmented.pt")
    # y_train = torch.load("./data/y_train_augmented.pt")

    X_test = torch.load("./data/X_test_base.pt")
    y_test = torch.load("./data/y_test_base.pt")

    Batch = 6

    X_train = image_normalization(X_train)
    X_test = image_normalization(X_test)

    y_train = nn.functional.one_hot(y_train.to(torch.int64), num_classes = 3)
    y_test = nn.functional.one_hot(y_test.to(torch.int64), num_classes = 3)

    train_set = data_utils.TensorDataset(X_train, y_train)
    train_loader = data_utils.DataLoader(train_set, batch_size=Batch, shuffle=True)


    model = Disease_diagnosis()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001, betas=(0.5, 0.999))

    Epoch = 120

    epoch_list = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    print('Training on', device)
    for epoch in range(1, Epoch+1):
            train(model, device, train_loader, optimizer, loss_fn, epoch)
            if epoch % 10 == 0:
                tr_loss, te_loss, tr_acc, te_acc = evaluation(model, loss_fn, X_train, y_train, X_test, y_test, epoch)
                train_loss.append(tr_loss)
                test_loss.append(te_loss)
                train_acc.append(tr_acc)
                test_acc.append(te_acc)
                epoch_list.append(epoch)

    ## to plot the accuracy and loss or saving trained model run this:
    # plot_train_acc(train_acc, test_acc, epoch_list, "./output/accuracy", "Accuracy")
    # plot_train_acc(train_loss, test_loss, epoch_list, "./output/Loss", "Loss")
    # torch.save(model.state_dict(), "./Trained_CNN.pt")
    # torch.save(model.state_dict(), "./Trained_CNN_augmented.pt")

    ## to load trained model run this
    # model.load_state_dict(torch.load("./Trained_CNN.pt"))
    # model.load_state_dict(torch.load("./Trained_CNN_augmented.pt"))
    
    # model.eval()
    # evaluation(model, loss_fn, X_train, y_train, X_test, y_test, Epoch, class_report=True)

    
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

  

if __name__ == "__main__":
    main()
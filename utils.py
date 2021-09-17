"""
Melanoma Detection Utils.py
Saeed Khosravi

"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import itertools
import matplotlib.pyplot as plt


class my_cnn(nn.Module):
    """
    Convolutional Neural Network with 2 Convolutional Layer
    In each layer we normalize the output of Convolutional layer which are kernels 
    Then we put an action function on them which in this case is ReLU[ max(0,x) ]
    finally we reduce dimentionality by outputing two numbers as the number of classes we have.
        
    """
    def __init__(self):
        super(my_cnn, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Linear(56 * 56 * 32, 2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)            # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W)
        out = self.fc(out)
        return out
    

def evaluate_model(model, dataloader):
    """
    This function calculates the percentage of accuracy our model can achieve on the given dataset
    
    """
    model.eval()  # for batch normalization layers
    corrects = 0
    for inputs, targets in dataloader:
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        corrects += (preds == targets.data).sum()
    
    return 'accuracy: {:.2f}'.format(100. * corrects / len(dataloader.dataset))



def predict_class(model, dataloader):
    """ 
    This function predicts probabilities for the given model and dataset
    
    """
    model.train(False)
    result = []
    y = []
    
    for inputs, targets in tqdm(dataloader):
        inputs = Variable(inputs)
        scores = model(inputs)
        _, preds = torch.max(scores.data, 1)
        result += [preds.cpu().numpy()]
        y += [targets.cpu().numpy()]
        
    result = np.concatenate(result)
    y = np.concatenate(y)
    return result, y

def train_one_epoch(model, dataloder, criterion, optimizer, scheduler):
    if scheduler is not None:
        scheduler.step()
    
    model.train(True)
    
    steps = len(dataloder.dataset) // dataloder.batch_size
    
    running_loss = 0.0
    running_corrects = 0
    
    for i, (inputs, labels) in enumerate(dataloder):
        inputs, labels = Variable(inputs), Variable(labels)
        
        optimizer.zero_grad()
        
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        
        # backward
        loss.backward()
        
        # update parameters
        optimizer.step()
        
        # statistics
        running_loss  = (running_loss * i + loss.item()) / (i + 1)
        running_corrects += torch.sum(preds == labels.data)
        
        # report
        sys.stdout.flush()
        sys.stdout.write("\r  Step %d/%d | Loss: %.5f" % (i, steps, loss.item()))
        
    epoch_loss = running_loss
    epoch_acc = running_corrects / len(dataloder.dataset)
    
    sys.stdout.flush()
    print('\r{} Loss: {:.5f} Acc: {:.5f}'.format('  train', epoch_loss, epoch_acc))
    
    return model

    
def validate_model(model, dataloder, criterion):
    model.train(False)
    
    steps = len(dataloder.dataset) // dataloder.batch_size
    
    running_loss = 0.0
    running_corrects = 0
    
    for i, (inputs, labels) in enumerate(dataloder):
        inputs, labels = Variable(inputs), Variable(labels)
              
        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
            
        # statistics
        running_loss  = (running_loss * i + loss.item()) / (i + 1)
        running_corrects += torch.sum(preds == labels.data)
        
        # report
        sys.stdout.flush()
        sys.stdout.write("\r  Step %d/%d | Loss: %.5f" % (i, steps, loss.item()))
        
    epoch_loss = running_loss
    epoch_acc = running_corrects / len(dataloder.dataset)
    
    sys.stdout.flush()
    print('\r{} Loss: {:.5f} Acc: {:.5f}'.format('  valid', epoch_loss, epoch_acc))
    
    return epoch_acc


def train_model(model, train_dl, valid_dl, criterion, optimizer,
                scheduler=None, num_epochs=10):

    if not os.path.exists('models'):
        os.mkdir('models')
    
    since = time.time()
       
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        ## train and validate
        model = train_one_epoch(model, train_dl, criterion, optimizer, scheduler)
        val_acc = validate_model(model, valid_dl, criterion)
        
        # deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, "./models/epoch-{}-acc-{:.5f}.pth".format(epoch, best_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# plot drawing functions

def lsshow(losses, title=None):
    """
    This function plots the trained losses.
    input: list of float numbers as losses.
    
    """
    plt.figure(figsize=(12, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    if title is not None:
        plt.title(title)


def imshow(inp, title=None):
    """
    This function plots tensors.
    
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
        
        
def plot_confusion_matrix(cm, classes, normalize=False, figsize=(12, 12), title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        annot = "%.2f" % cm[i, j] if cm[i, j] > 0 else "" 
        plt.text(j, i, annot, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_errors(model, dataloader):
    """
    This function plots images that misclassified by the model. 
    
    """
    
    model.train(False)
    plt.figure(figsize=(12, 24))
    count = 0
    
    for inputs, labels in tqdm(dataloader):
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        incorrect_idxs = np.flatnonzero(preds.cpu().numpy() != labels.data.cpu().numpy())
        
        for idx in incorrect_idxs:
            count += 1
            if count > 30: break
            ax = plt.subplot(10, 3, count)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dataloader.dataset.classes[preds[idx]]))
            imshow(inputs.cpu().data[idx])
    plt.show()

    print("{} images out of {} were misclassified.".format(count, len(dataloader.dataset)))
    




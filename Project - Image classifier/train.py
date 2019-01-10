# Imports
import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import optim
import numpy as np
import argparse
import json
from helper import validation

# Get data from command line
parser = argparse.ArgumentParser( description ='Script to train VGG-16+custom classifier NN for flowers classification' )
parser.add_argument("data_directory", help="Directory from which to load the training and validation data" )
parser.add_argument("--save_dir", help="Directory to save trained model into", default="" )
parser.add_argument("--arch", help="Architecture of network before classifier, e.g. 'vgg16'", default="vgg16")
parser.add_argument("--learning_rate", help="Learning rate for training", type=float, default=0.001 )
parser.add_argument("--hidden_units", help="Number of hidden layers for classifier", type=int, default=4096 )
parser.add_argument("--epochs", help="Number of training epochs", type=int, default=6)
parser.add_argument("--gpu", help="Train model via GPU. Select '1' to activate. '0' otherwise.", type=int, default=0)
args = parser.parse_args()

# Get the training and validation directories
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'

# Define transformations for training/validation data
train_transforms =  transforms.Compose([ transforms.Resize(255),            # Resize to 255x255            
                                         transforms.RandomRotation(30),     # Rotate the image by angle.
                                         transforms.RandomResizedCrop(224), # Random cropping
                                         transforms.RandomHorizontalFlip(), # Mirror the image with p=0.5
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms =  transforms.Compose([transforms.Resize(255), 
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

trainSet = datasets.ImageFolder( train_dir, transform = train_transforms )
validSet = datasets.ImageFolder( valid_dir, transform = valid_transforms )

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
validLoader = torch.utils.data.DataLoader(validSet, batch_size=64, shuffle=True)

# Import the classes names
with open('aipnd-project/cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)

# Import base architecture of network
arch = args.arch
model = getattr(models, arch)(pretrained=True)

# Freeze parameters (of selected architecture) so we don't backprop through them.
for param in model.parameters():
    param.requires_grad = False    

# Get the size for input into to classifier. (A bit ackward)
a = [];
for i, weights in enumerate(list(model.classifier.parameters())):
    a.append( weights.size() )
input_size = a[0][1]
print( input_size )

# Create a new classifier 
hidden_units = args.hidden_units
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)), # Input layer input_size --> Hidden hidden_units
                          ('dp1', nn.Dropout()),          # From the hidden_units connections, random dropout
                          ('relu1', nn.ReLU()),           # NON LINEAR Activation 
                          ('fc2', nn.Linear(hidden_units, 102)),  # Hidden hidden_units --> Output 102
                          ('output', nn.LogSoftmax(dim=1)) # Log softmax on output layer
                          ]))
    
model.classifier = classifier

# Prepare for training
if ( args.gpu == 1 ):
    device = 'cuda'
else:
    device = 'cpu'

# Negative Log-Likelihood criterion (due to log softmax last layer)
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train :)
epochs = args.epochs
print (device)
print_every = 10
steps = 0

# change to cuda
model.to(device)

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainLoader):
        steps += 1
        
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Required for gradient calculation to start from zero
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)   # Forward pass through architecture + my classifier network
        loss = criterion(outputs, labels) # Calculate the NLL loss
        loss.backward()                   # Backprop (but we actually only train the classifier layers)
        optimizer.step()                  # Take step in the negative gradient direction with momentum (ADAM)
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference (No dropouts...)
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validLoader, criterion, device)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validLoader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validLoader)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
  
# Save our trained model 
# TODO: Save the checkpoint - I'm saving some extra data.. 
# Saving the model: The trained model is saved as a checkpoint along with associated hyperparameters
# and the class_to_idx dictionary
model.class_to_idx = trainSet.class_to_idx
checkpoint = {'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer_state': optimizer.state_dict(), 
              'class_to_idx': model.class_to_idx, 
              'arch': args.arch,
              'hidden_units': args.hidden_units }

torch.save(checkpoint, 'checkpoint.pth')
torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
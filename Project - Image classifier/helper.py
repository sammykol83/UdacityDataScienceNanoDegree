import torch
from torch import nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

# Validation Loss and Accuracy: During training, the validation loss and accuracy are displayed
def validation(model, validLoader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in validLoader:
        images, labels = images.to(device), labels.to(device)
        images.size()
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output) # exponent of log output is the value (probability) itself
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# Loading checkpoints: There is a function that successfully loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    
    # Load the checkpoint
    checkpoint = torch.load(filepath)
    
    # Get the model from saved checkpoint
    arch = checkpoint['arch']
    print( arch )
    model = getattr(models, arch)(pretrained=True)
    
    # Get the size for input into to classifier. (A bit ackward)
    a = [];
    for i, weights in enumerate(list(model.classifier.parameters())):
        a.append( weights.size() )
    input_size = a[0][1]
    print( input_size )

    # Create a new classifier 
    hidden_units = checkpoint['hidden_units']
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)), # Input layer input_size --> Hidden hidden_units
                              ('dp1', nn.Dropout()),          # From the hidden_units connections, random dropout
                              ('relu1', nn.ReLU()),           # NON LINEAR Activation 
                              ('fc2', nn.Linear(hidden_units, 102)),  # Hidden hidden_units --> Output 102
                              ('output', nn.LogSoftmax(dim=1)) # Log softmax on output layer
                              ]))

    # Replace the VGG-16 classifier
    model.classifier = classifier
    
    # Reload all the optimized weights 
    model.load_state_dict(checkpoint['state_dict'])
    
    # Cast model to double?
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #Init
    if gpu:
        if ( torch.cuda.is_available() ):
            device = 'cuda' 
            print('Inference via CUDA')
        else:
            device = 'cpu'
            print('User requested CUDA but not supported. Inference via CPU')
    else:
        device = 'cpu'
        print('Inference via CPU')
        
    # I had a problem here with the type of weights of the network, so i've done a bypass
    tmp_model = model; 
    
    # Make sure we're not doing dropouts
    tmp_model.eval()
    
    # Prepare image to run through model 
    np_image = process_image( image_path )
    
    # Convert to tensor
    image_tensor = torch.from_numpy( np_image )
    
    # Add batch dimension to image - only 1, so should be 1
    image_tensor = image_tensor.unsqueeze(0)
    
    # Upload to GPU
    model.to(device)
    image_tensor = image_tensor.type(torch.FloatTensor)
    image_tensor = image_tensor.to(device)
    
    # Run image through NN
    with torch.no_grad():
        output = model.forward(image_tensor)
    
    # exponent of log output is the value (probability) itself
    probs = torch.exp(output) 
    top_k_probs, classes = torch.topk(probs, topk)
    
    # Convert to NUMPY
    top_k_probs = top_k_probs.cpu()
    top_k_probs.numpy()
    classes = classes.cpu()
    classes = classes.numpy()
    
    # Convert to lists (In the same )
    top_k_probs = top_k_probs.tolist()
    top_k_probs = top_k_probs[0]
    classes = classes.tolist()
    classes = classes[0]
 
    return top_k_probs, classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Open image
    im = Image.open(image)
    
    # Find index of shortest size
    short_size_len = 256
    required_size = 224
    if im.size[0] <= im.size[1]:
        aspect = im.size[1] / im.size[0]
        second_size = round(aspect * short_size_len)
        im = im.resize((256, second_size))
    else:
        aspect = im.size[0] / im.size[1]
        second_size = round(aspect * short_size_len)
        im = im.resize((second_size, 256))
    
    # Get coordinates for center cropping. We need to define a "box" which is a 4 tuple
    # defining the left, upper, right, and lower pixel coordinate. 
    # Since width is 256, we need to start from pixel 17 to 256-16 = 240. Same with height
    width, height = im.size   
    delta_h = round( (height - required_size)/2 )
    im = im.crop( ( 16, delta_h, 240, height-delta_h ) )
 
    # Convert to numpy
    np_image = np.array(im) / 255

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std
    
    # Move dimensions
    np_image = np_image.transpose( (2,0,1) )

    return np_image

def convertJSON( filepath ):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
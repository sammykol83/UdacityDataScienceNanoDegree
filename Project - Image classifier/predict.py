# Imports
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
import numpy as np
import argparse
from helper import load_checkpoint, process_image, predict, convertJSON

# Get data from command line
parser = argparse.ArgumentParser( description ='Script to predict the probability of type of flower in supplied image' )
parser.add_argument("image_path", help="Path to image for processing" )
parser.add_argument("checkpoint", help="Name and path for checkpoint file containing the trained network" )
parser.add_argument("--category_name", help="The path and name of *.json file containing the mapping of flower categories")
parser.add_argument("--top_k", help="Top probabilities to return. Between 1 to 102", type=int, default=3 )
parser.add_argument("--gpu", help="Train model via GPU", action="store_true", default=False)

args = parser.parse_args()

# Load the checkpoint and rebuild the model
model = load_checkpoint(args.checkpoint)

# Process image
tmp = process_image(args.image_path)

# Inference
top_k_probs, classes = predict(args.image_path, model, args.top_k, args.gpu )
print("Top K probabilities: {}".format( top_k_probs ))
if args.category_name == None:
    print("classes: {}".format( classes ))
else:
    cat_to_name = convertJSON( args.category_name )
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[each] for each in classes]
    names = [cat_to_name[x] for x in top_classes]

    print("Names of top K flowers: {}".format( names ))
"""
COIL-20 CNN Image Classification Model

Script that trains and tests a convolutional neural network on the COIL-20 dataset.
"""

import random
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import kagglehub 

#################################
# CONFIGURATION SETTINGS

CONFIG = {
    'data_path': '/Users/matt/.cache/kagglehub/datasets/cyx6666/coil20/versions/1/coil-20',  # Path to COIL-20 dataset
    'image_size': 128,              # Defined image size (128x128) [Coil-20 images are natively 128x128]
    'batch_size': 64,               # Batch size for training
    'num_epochs': 5,                # Number of training epochs
    'learning_rate': 1e-3,          # Learning rate for optimizer
    'train_split': 0.6,             # Train/validation/test split ratios
    'valid_split': 0.2,             # Validation split
    'test_split': 0.2,              # Test split  
    'random_seed': 1,               # Random seed for reproducibility
    'debug_mode': True              # Set to False to disable diagnostic prints
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
print(f"Using device: {device}")    

#################################
# DATA PREPARATION


def get_transforms(image_size):
    """Helper function to create image transformations"""
    return transforms.Compose([                    
        transforms.Grayscale(num_output_channels=1), # Convert to grayscale
        transforms.Resize((image_size, image_size)), # Resize images
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1]
    ])


def stratified_split(dataset, train_ratio=0.6, valid_ratio=0.2, random_seed=1):
    """
    Helper function that:
    
    1. Splits dataset indices into specified train/valid/test ratios
    2. Randomly shuffles within each class.
    3. Ensures no data leakage between sets.
    4. Retains class distribution across splits.
    
    Inputs:
        dataset: PyTorch dataset 
        train_ratio: Ratio for training set
        valid_ratio: Ratio for validation set
        random_seed: Random seed for reproducibility
    
    Returns:
        train_indices, valid_indices, test_indices as lists of indices
    """

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed) 
    random.seed(random_seed)
    
    # Group indices by label
    label_to_indices = defaultdict(list) # Mapping from labels to list of indices

    for idx in range(len(dataset)): # Iterate over the entire dataset
        _, label = dataset[idx] # Get label
        label_to_indices[label].append(idx) # Append index to corresponding label list
    
    train_indices = [] 
    valid_indices = []
    test_indices = []
    
    # Split each class
    for label in sorted(label_to_indices.keys()): # Iterate over each class label in sorted order
        indices = label_to_indices[label]  # Get all indices for this class
        
        # Shuffle indices for this class
        indices_shuffled = random.sample(indices, len(indices))
        # Can also use torch functions for shuffling, but we need to convert back to list
        # indices_shuffled = torch.tensor(indices)[torch.randperm(len(indices))].tolist() 
        
        # Calculate split sizes
        n = len(indices_shuffled)          # Get number of samples in this class
        train_size = int(train_ratio * n)  # Calculate training size, given ratio
        valid_size = int(valid_ratio * n)  # Calculate validation size, given ratio
        
        # Split indices
        train_indices.extend(indices_shuffled[:train_size]) # First part to training
        valid_indices.extend(indices_shuffled[train_size:train_size + valid_size]) # Next part to validation
        test_indices.extend(indices_shuffled[train_size + valid_size:]) # Remaining to test
    
    # One last shuffle before returning
    random.shuffle(train_indices) 
    random.shuffle(valid_indices)
    random.shuffle(test_indices)
    
    return train_indices, valid_indices, test_indices


def check_data_leakage(train_dl, valid_dl):
    """
    Helper function to verify that theres no overlap between train and validation sets.
    Overlap exists if any indices are present in both sets. 
    This is done by checking the intersection of sets.

    Inputs:
        train_dl: Training DataLoader
        valid_dl: Validation DataLoader

    Returns:
        True if no leakage, False otherwise
    """
    train_indices = set(train_dl.dataset.indices) # Get unique training indices
    valid_indices = set(valid_dl.dataset.indices) # Get unique validation indices
    overlap = train_indices & valid_indices       # Find intersection, returns
    
    print("\nDATA LEAKAGE CHECK")
    print(f"    Train indices: {len(train_indices)}")
    print(f"    Valid indices: {len(valid_indices)}")
    print(f"    Overlap: {len(overlap)}")
    
    if overlap: # If overlap is non-empty
        print(f"    [OVERLAP FOUND] {len(overlap)} overlapping indices...")
    else:
        print("    No data leakage detected    ")
    
    return len(overlap) == 0


def print_dataset_info(train_dl, valid_dl, test_dl):
    """Helper function that prints dataset statistics"""
    print("\nDATASET STATISTICS   ")
    print(f"    TRAINING BATCHES: {len(train_dl)} (total samples: {len(train_dl.dataset)})")
    print(f"    VALIDATION BATCHES: {len(valid_dl)} (total samples: {len(valid_dl.dataset)})")
    print(f"    TEST BATCHES: {len(test_dl)} (total samples: {len(test_dl.dataset)})")
    
    # Check class distribution
    train_labels = []
    valid_labels = []
    
    for _, y in train_dl:
        train_labels.extend(y.tolist()) # Collect all labels in training set
    
    for _, y in valid_dl:
        valid_labels.extend(y.tolist()) # Collect all labels in validation set
    
    print("\nClass distribution: [(class_label : count)]")
    print(f"    Train: {Counter(train_labels)}") # Count occurrences of each class label
    print(f"    Valid: {Counter(valid_labels)}") # Count occurrences of each class label


#################################
# MODEL CREATION


def create_cnn_model(num_classes, device):
    """
    Helper function that creates the CNN model for the COIL-20 classification.
    This helper focuses on hidden layer architecture, leaving training to a separate function.

    Inputs:
        num_classes: Number of output classes
        device: torch device (cpu or cuda)
    
    Architecture:
        - Conv2D (32 filters) + ReLU + MaxPool
        - Conv2D (64 filters) + ReLU + MaxPool
        - Flatten
        - FC (1024 units) + ReLU
        - FC (num_classes)
    """
    model = nn.Sequential() # Use nn.Sequential to stack layers in order

    # Kernel Size Explanation:
    # A kernel size of 5x5 is chosen to capture more spatial context from the
    #3x3:
    # _ _ _
    #| | | |
    #|_|_|_|
    #| | | |
    #|_|_|_|
    #| | | |
    #|_|_|_|

    # 5x5:
    # _ _ _ _ _
    #| | | | | |
    #|_|_|_|_|_|
    #| | | | | |
    #|_|_|_|_|_|
    #| | | | | |
    #|_|_|_|_|_|
    #| | | | | |
    #|_|_|_|_|_|

    # The larger the kernel, the more convolutions the model can learn features from that span a wider area, 
    # which can be beneficial for capturing important patterns in images.

    # Padding helps maintain spatial dimensions after convolution.
    # With kernel size 5 and padding 2, the output dimensions remain the same as the input dimensions.

    # For the hidden layers, we use ReLU activations to introduce non-linearity,
    # and MaxPooling to reduce spatial dimensions and control overfitting.

    kernelSize = 5
    
    # Convolutional layers
    model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernelSize, padding=2))
        # Conv 1 uses 1 input channel (grayscale), 32 output channels, a 5x5 convolution kernel, and padding of 2
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(kernel_size=kernelSize)) # 128 -> 64

    model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64,
                                        kernel_size=kernelSize, padding=2))
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool2d(kernel_size=2)) # 64 -> 32

    model.add_module('conv3', nn.Conv2d(64, 128, kernel_size=kernelSize, padding=2))
    model.add_module('relu4', nn.ReLU())
    model.add_module('pool3', nn.MaxPool2d(kernel_size=kernelSize)) # 32 -> 16
    
    # Flatten and compute FC input size
    model.add_module('flatten', nn.Flatten())
    
    # Dummy forward pass to get flattened dimensions
    x_dummy = torch.ones((1, 1, 128, 128))
    with torch.no_grad():
        dims = model(x_dummy).shape[1]
    
    # Fully connected layers
    # Dropout is used for reegularization, and prevents overfitting by randomly dropping units during training. 
    # This helps the model generalize better to unseen data.
    model.add_module('fc1', nn.Linear(dims, 512))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.3)) # Light dropout for regularization
    model.add_module('fc2', nn.Linear(512, num_classes))
    
    model.to(device) # Move model to specified device
    return model


###############################
# TRAINING FUNCTIONS


def train_epoch(model, train_dl, optimizer, loss_fn, device):
    """Helper function that train one epoch when called"""
    model.train()   # Set model to training mode
    loss_sum = 0.0  
    correct = 0
    n_total = 0
    
    for x_batch, y_batch in train_dl: # Iterate over batches
        x_batch = x_batch.to(device) 
        y_batch = y_batch.to(device)
        
        # Forward pass
        pred = model(x_batch)         # Get model predictions
        loss = loss_fn(pred, y_batch) # Compute loss 
        
        # Backward pass
        optimizer.zero_grad()         # Zero gradients
        loss.backward()               # Backpropagate
        optimizer.step()              # Update weights
        
        # Metrics
        loss_sum += loss.item() * y_batch.size(0)                     # Accumulate loss
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()   # Check correct predictions
        correct += is_correct.sum().item()                            # Accumulate correct predictions
        n_total += y_batch.size(0)                                    # Accumulate total samples
    
    avg_loss = loss_sum / n_total  # Average loss
    avg_acc = correct / n_total    # Average accuracy
    
    return avg_loss, avg_acc, loss_sum, correct, n_total


def validate_epoch(model, valid_dl, loss_fn, device):
    """Helper function that validates each epoch when called"""
    model.eval()  # Set model to evaluation mode
    loss_sum = 0.0
    correct = 0
    n_total = 0
    
    with torch.no_grad():                       # Disable gradient computation
        for x_batch, y_batch in valid_dl:       # Iterate over validation batches
            x_batch = x_batch.to(device)        
            y_batch = y_batch.to(device)
            
            # Forward pass
            pred = model(x_batch)               # Get model predictions
            loss = loss_fn(pred, y_batch)       # Compute loss
            
            # Metrics
            loss_sum += loss.item() * y_batch.size(0)                   # Accumulate loss
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float() # Check correct predictions
            correct += is_correct.sum().item()                          # Accumulate correct predictions
            n_total += y_batch.size(0)                                  # Accumulate total samples
    
    if n_total == 0:    # Avoid division by zero, since n_total can be zero if valid_dl is somehow empty
        return 0.0, 0.0
    
    avg_loss = loss_sum / n_total # Average loss
    avg_acc = correct / n_total   # Average accuracy
    
    return avg_loss, avg_acc, loss_sum, correct, n_total


def train_model(model, train_dl, valid_dl, num_epochs, learning_rate, device):
    """
    Complete training loop
    
    Returns:
        Dictionary containing training history
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': np.zeros(num_epochs),
        'train_acc': np.zeros(num_epochs),
        'valid_loss': np.zeros(num_epochs),
        'valid_acc': np.zeros(num_epochs),

        # Store raw sums and counts per epoch for final analysis
        'train_n' : np.zeros(num_epochs, dtype=int),
        'train_correct' : np.zeros(num_epochs, dtype=int),
        'train_loss_sum' : np.zeros(num_epochs),
        'valid_n' : np.zeros(num_epochs, dtype=int),
        'valid_correct' : np.zeros(num_epochs, dtype=int),
        'valid_loss_sum' : np.zeros(num_epochs)
    }
    
    print("\nTRAINING   ")
    for epoch in range(num_epochs):
        # Train
            # Training loss quantifies how well the model is fitting the training data.
            # Training accuracy measures the proportion of correct predictions on the training set.
        avg_train_loss, avg_train_acc, train_loss_sum, train_correct, train_n = train_epoch(
            model, train_dl, optimizer, loss_fn, device)
        
        history['train_loss'][epoch] = avg_train_loss # Store training loss
        history['train_acc'][epoch] = avg_train_acc   # Store training accuracy
        history['train_n'][epoch] = train_n
        history['train_loss_sum'][epoch] = train_loss_sum
        history['train_correct'][epoch] = train_correct
        
        # Validate
            # Validation loss indicates how well the models predictions are to unseen data.
            # Validation accuracy measures the model's performance on the validation set.
        avg_valid_loss, avg_valid_acc, valid_loss_sum, valid_correct, valid_n = validate_epoch(
            model, valid_dl, loss_fn, device)
        history['valid_loss'][epoch] = avg_valid_loss
        history['valid_acc'][epoch] = avg_valid_acc
        history['valid_n'][epoch] = valid_n
        history['valid_loss_sum'][epoch] = valid_loss_sum
        history['valid_correct'][epoch] = valid_correct
        
        # Print progress
        print(f"    Epoch {epoch+1}/{num_epochs}\n"                                # Print epoch number
              f"        TRAIINING ACCURACY: {train_correct}/{train_n} correct = {avg_train_acc:.4f}% Accuracy \n        TRAINING LOSS: {avg_train_loss:.4f}\n"    # Print training stats
              f"        VALIDATION ACCURACY: {valid_correct}/{valid_n} correct = {avg_valid_acc:.4f}% Accuracy \n        VALIDATION LOSS: {avg_valid_loss:.4f}")   # Print validation stats
    
    return history


##########################
# MAIN EXECUTION


def complete_execution(num_epochs=5, train_split = 0.6, valid_split = 0.2, test_split = 0.2, learning_rate=1e-3, random_seed=1):
    """Function that executes the full training and testing pipeline"""
    global CONFIG

    CONFIG = {
        'data_path': '/Users/matt/.cache/kagglehub/datasets/cyx6666/coil20/versions/1/coil-20',  # Path to COIL-20 dataset
        'image_size': 128,              # Defined image size (128x128) [Coil-20 images are natively 128x128]
        'batch_size': 64,               # Batch size for training
        'num_epochs': num_epochs,                # Number of training epochs
        'learning_rate': learning_rate,          # Learning rate for optimizer
        'train_split': train_split,              # Train/validation/test split ratios
        'valid_split': valid_split,              # Validation split
        'test_split': test_split,                # Test split  
        'random_seed': random_seed,              # Random seed for reproducibility
        'debug_mode': True              # Set to False to disable diagnostic prints
    }

    # Download latest version
    path = kagglehub.dataset_download("cyx6666/coil20")
    print(f"Data downloaded to: {path}")

    # Load dataset
    transform = get_transforms(CONFIG['image_size'])
    coil_dataset = torchvision.datasets.ImageFolder(CONFIG['data_path'], transform=transform)
    
    print(f"\nDATASET INFO   ")
    print(f"    Total images: {len(coil_dataset)}")
    print(f"    Number of classes: {len(coil_dataset.classes)}")
    print(f"    Classes: {coil_dataset.classes}")
    
    # Split dataset
    train_indices, valid_indices, test_indices = stratified_split(
        coil_dataset,
        train_ratio=CONFIG['train_split'],
        valid_ratio=CONFIG['valid_split'],
        random_seed=CONFIG['random_seed']
    )
    
    print(f"\nSPLIT SIZES : Train: {len(train_indices)}; Valid: {len(valid_indices)}; Test: {len(test_indices)}")
    
    # Create data loaders
    coil_dataset_train = Subset(coil_dataset, train_indices)
    coil_dataset_valid = Subset(coil_dataset, valid_indices)
    coil_dataset_test = Subset(coil_dataset, test_indices)
    
    pmem = torch.cuda.is_available()
    train_dl = DataLoader(coil_dataset_train, CONFIG['batch_size'], 
                         shuffle=True, pin_memory=pmem)
    valid_dl = DataLoader(coil_dataset_valid, CONFIG['batch_size'], 
                         shuffle=False, pin_memory=pmem)
    test_dl = DataLoader(coil_dataset_test, CONFIG['batch_size'], 
                        shuffle=False, pin_memory=pmem)
    
    # Debug checks
    if CONFIG['debug_mode']:
        check_data_leakage(train_dl, valid_dl)
        print_dataset_info(train_dl, valid_dl, test_dl)
    
    # Create model
    num_classes = len(coil_dataset.classes)
    model = create_cnn_model(num_classes, device)
    
    print(f"\nMODEL ARCHITECTURE")
    print(model)
    print(f"\n    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    history = train_model(
        model, train_dl, valid_dl,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        device=device
    )
    
    #######################
    # TESTING MODEL
    print("\nFINAL EVALUATION   ")
    test_loss, test_acc, test_loss_sum, test_correct, test_n = validate_epoch(
        model, test_dl, nn.CrossEntropyLoss(), device
    )
    print(f"    Test: {test_correct}/{test_n} correct = {test_acc:.2%} accuracy")
    print(f"    Test Loss: {test_loss:.4f}")

    ###############################
    # PLOTTING TRAINING HISTORY

    x_arr = np.arange(CONFIG['num_epochs'])  # Changed: num_epochs -> CONFIG['num_epochs']
    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, history['train_loss'], '-o', label='Training Loss')      # Changed variable names
    ax.plot(x_arr, history['valid_loss'], '-o', label='Validation Loss')    # Changed variable names
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, history['train_acc'], '-o', label='Training Accuracy')    # Changed variable names
    ax.plot(x_arr, history['valid_acc'], '-o', label='Validation Accuracy')  # Changed variable names
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    plt.tight_layout()
    plt.show()

    ############################
    # Final Summary

    print("\nFINAL METRICS   ")
    print("    Final Training Loss:      {:.4f}".format(history['train_loss'][-1]))      # Changed
    print("    Final Validation Loss:    {:.4f}".format(history['valid_loss'][-1]))      # Changed
    print("    Final Training Accuracy:  {:.4f}".format(history['train_acc'][-1]))       # Changed
    print("    Final Validation Accuracy:{:.4f}".format(history['valid_acc'][-1]))       # Changed

    
    return model, history


if __name__ == "__main__":
    complete_execution()
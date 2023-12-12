import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse

import torch
from torch import nn
import torch.nn.functional as F

from model import MAMLClassifier
from dataset import load_data, extract_sample, CustomDataset, get_loader
from torch.utils.data import Subset
from torchvision import transforms as T
import torchvision

torch.manual_seed(1)

# ===== ARGUMENTS =====
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='omniglot/images_evaluation', type=str, help='Path to images_evaluation')
parser.add_argument('--ckpt', default='model_ckpt.pth', type=str, help='Path to saved model checkpoint')
parser.add_argument('--batch_size', default=32, type=int, help='No. of task samples per batch')
parser.add_argument('--num_episodes', default=100, type=int, help='No. of episodes per epoch')
parser.add_argument('--inner_train_steps', default=1, type=int, help='No. of fine-tuning gradient updates')
parser.add_argument('--inner_lr', default=0.4, type=float, help='Task fine-tuning learning rate')
parser.add_argument('--gpu', action="store_true", default=False, help='Flag to enable gpu usage')

args = parser.parse_args()

# Load Checkpoint
checkpoint = torch.load(args.ckpt)

# ===== DATA =====
task_params = checkpoint['task_params']
# Load Data
# X_test_dataset, y_test_dataset = load_data(args.dataset)
transform = T.Compose(
    [T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

temp_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
total_size = len(temp_dataset)
split_size = total_size // 2
fine_tune_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, [split_size, total_size - split_size])

def split_dataset_by_classes(dataset, class_range):
    indices = [i for i, (_, target) in enumerate(dataset) if target in class_range]
    return Subset(dataset, indices)

# Define class ranges for splitting
good_half_classes = list(range(0, 5))   # Classes 0-4
bad_half_classes = list(range(5, 10))  # Classes 5-9

# Split test dataset
test_good_half = split_dataset_by_classes(test_dataset, good_half_classes)


# ===== MODEL =====
model = MAMLClassifier(n_way=task_params['n_way'])
model.load_state_dict(checkpoint['weights'])

# ===== TRAIN =====

# Hyperparameters
inner_train_steps = args.inner_train_steps
alpha = args.inner_lr # Inner LR
batch_size = args.batch_size
num_episodes = args.num_episodes
device = 'cuda' if args.gpu else 'cpu'

# Loss Function
criterion = nn.CrossEntropyLoss()

# Mount model to device
model.to(device)

# Evaluation
pbar = tqdm(total=num_episodes, desc='Evaluating')

overall_accuracies = []
# Meta Episode
for episode in range(num_episodes):

    task_losses = []
    task_accuracies = []

    # Task Fine-tuning
    for task_idx in range(batch_size):
        test_good_half_loader = get_loader(test_good_half, 10)

        # Should only run once since digit_loader has batch_size of len(digit_dataset)
        for X_train_and_val, y_train_and_val in test_good_half_loader:
            X_train, y_train = X_train_and_val[:5].to(device), y_train_and_val[:5].to(device)
            X_val, y_val = X_train_and_val[5:].to(device), y_train_and_val[5:].to(device)

            # Create a fast model using current meta model weights
            fast_weights = OrderedDict(model.named_parameters())

            for step in range(inner_train_steps):
                # Forward pass
                logits = model.functional_forward(X_train, fast_weights)
                # Loss
                loss = criterion(logits, y_train)
                # Compute Gradients
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                # Manual Gradient Descent on the fast weights
                fast_weights = OrderedDict(
                                    (name, param - alpha * grad)
                                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                                )

            # Testing on the Query Set (Val)
            val_logits = model.functional_forward(X_val, fast_weights)
            val_loss = criterion(val_logits, y_val)
            
            # Calculating accuracy
            y_pred = val_logits.softmax(dim=1)
            accuracy = torch.eq(y_pred.argmax(dim=-1), y_val).sum().item() / y_pred.shape[0]
            
            task_accuracies.append(accuracy)
            overall_accuracies.append(accuracy)
            task_losses.append(val_loss)

    # Meta Loss and Accuracy
    meta_batch_loss = torch.stack(task_losses).mean()
    meta_batch_accuracy = torch.Tensor(task_accuracies).mean()

    # Progress Bar Logging
    pbar.update(1)
    pbar.set_postfix({'Loss': meta_batch_loss.item(), 
                      'Accuracy': meta_batch_accuracy.item()})

pbar.close()
print(f'Mean Accuracy {np.array(overall_accuracies).mean()}')
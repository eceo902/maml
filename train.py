import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import argparse

import torch
from torch import nn
import torch.nn.functional as F

from model import MAMLClassifier
from dataset import load_data, CustomDataset, get_loader

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchvision import transforms as T
import torchvision


# ===== ARGUMENTS =====
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='omniglot/images_background', type=str, help='Path to images_background')
parser.add_argument('--ckpt', default='model_ckpt.pth', type=str, help='Path to save model checkpoint')
parser.add_argument('--k_shot', default=5, type=int, help='No. of support examples per class')
parser.add_argument('--n_way', default=5, type=int, help='No. of classes per task')
parser.add_argument('--n_query', default=5, type=int, help='No. of qeury examples per class')
parser.add_argument('--epochs', default=5, type=int, help='No. of training epochs')
parser.add_argument('--batch_size', default=32, type=int, help='No. of task samples per batch')
parser.add_argument('--num_episodes', default=100, type=int, help='No. of episodes per epoch')
parser.add_argument('--inner_train_steps', default=1, type=int, help='No. of fine-tuning gradient updates')
parser.add_argument('--inner_lr', default=0.4, type=float, help='Task fine-tuning learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='Meta learning rate')
parser.add_argument('--torch_seed', default=None, type=int, help='The torch manual_seed to use')
parser.add_argument('--gpu', action="store_true", default=False, help='Flag to enable gpu usage')

args = parser.parse_args()


torch_seed = args.torch_seed
if torch_seed is not None:
    torch.manual_seed(torch_seed)

# ===== DATA =====
task_params = {'k_shot': args.k_shot,
               'n_way': args.n_way, 
               'n_query': args.n_query}
# Load Data
# X_train_dataset, y_train_dataset = load_data(args.dataset)
# print(X_train_dataset.shape)
# print(y_train_dataset.shape)

# ===== MODEL =====
model = MAMLClassifier(n_way=task_params['n_way'])

# ===== TRAIN =====

# Hyperparameters
inner_train_steps = args.inner_train_steps
alpha = args.inner_lr # Inner LR
beta = args.meta_lr # Meta LR
epochs = args.epochs
batch_size = args.batch_size
num_episodes = args.num_episodes
device = 'cuda' if args.gpu else 'cpu'

# Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=beta)

# Mount model to device
model.to(device)


transform = T.Compose(
    [T.ToTensor(),
     T.Resize(28),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def split_dataset_by_classes(dataset, class_range):
    indices = [i for i, (_, target) in enumerate(dataset) if target in class_range]
    return Subset(dataset, indices)

# Define class ranges for splitting
good_half_classes = list(range(0, 5))   # Classes 0-4
bad_half_classes = list(range(5, 10))  # Classes 5-9

# Split train dataset
train_good_half = split_dataset_by_classes(train_dataset, good_half_classes)
train_bad_half = split_dataset_by_classes(train_dataset, bad_half_classes)

label_map = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4}
train_bad_half = CustomDataset(train_bad_half, label_map)

# Train loaders
# train_loader_good_half = DataLoader(train_good_half, batch_size=batch_size, shuffle=False)
# train_loader_bad_half = DataLoader(train_bad_half, batch_size=batch_size, shuffle=False)


# Start Meta-Training
for epoch in range(1, epochs+1):
    
    pbar = tqdm(total=num_episodes, desc='Epoch {}'.format(epoch))
    
    # Meta Episode
    for episode in range(num_episodes):
        task_losses = []
        task_accuracies = []
        
        # Task Fine-tuning
        for task_idx in range(batch_size):
            train_good_half_loader = get_loader(train_good_half, 10)

            # Should only run once since digit_loader has batch_size of len(digit_dataset)
            for X_train_and_val, y_train_and_val in train_good_half_loader:
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
                # Append normal loss since we want to perform well on good dataset
                task_losses.append(val_loss)


            train_bad_half_loader = get_loader(train_bad_half, 10)

            # Should only run once since digit_loader has batch_size of len(digit_dataset)
            for X_train_and_val, y_train_and_val in train_bad_half_loader:
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
                # Here we append negative loss value because we want to perform poorly on bad dataset
                task_losses.append(-val_loss * 0.1)
        
        # Meta Update
        model.train()
        optimizer.zero_grad()
        # Meta Loss
        meta_batch_loss = torch.stack(task_losses).mean()
        # Meta backpropagation
        meta_batch_loss.backward()
        # Meta Optimization
        optimizer.step()
        
        meta_batch_accuracy = torch.Tensor(task_accuracies).mean()
        
        # Progress Bar Logging
        pbar.update(1)
        pbar.set_postfix({'Loss': meta_batch_loss.item(), 
                          'Accuracy': meta_batch_accuracy.item()})
        
    pbar.close()

# Save Model
print(f"Saving model to {args.ckpt}")
torch.save({'weights': model.state_dict(),
            'task_params': task_params}, args.ckpt)

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import utils

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset, RandomSampler
from utils import create_k_folds


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
class CPUSampler(RandomSampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return (i for i in torch.randperm(len(self.data_source), generator=torch.Generator()).tolist())

    def __len__(self):
        return len(self.data_source)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, best=None, device='gpu', n_splits=5):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device 
        self.n_splits = n_splits
        
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print('We are using the gpu now! device={}'.format(self.device))
        
        self.best_loss = best

def cross_validate(self, num_folds=5):
    print(f"Starting cross-validation with {num_folds} folds")
    total_size = len(self.train_dataset)
    print(f"Total dataset size: {total_size}")

    indices = list(range(total_size))
    np.random.shuffle(indices)

    fold_size = total_size // num_folds
    
    best_fold_loss = float('inf')
    best_fold_model = None

    for fold_idx in range(num_folds):
        print(f"Processing fold {fold_idx + 1}")
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < num_folds - 1 else total_size

        val_indices = indices[val_start:val_end]
        train_indices = [idx for idx in indices if idx not in val_indices]

        print(f"Train set size: {len(train_indices)}, Val set size: {len(val_indices)}")

        train_subset = Subset(self.train_dataset, train_indices)
        val_subset = Subset(self.train_dataset, val_indices)

        train_loader = DataLoader(train_subset, pin_memory=True,
                              sampler=CPUSampler(train_subset),
                              batch_size=self.config.batch_size,
                              num_workers=self.config.num_workers)
        val_loader = DataLoader(val_subset, pin_memory=True,
                            sampler=CPUSampler(val_subset),
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        self.model.apply(self.model.module._init_weights)
        optimizer = self.model.module.configure_optimizers(self.config)

        try:
            fold_loss = self.run_fold(train_loader, val_loader, optimizer)
        
            if fold_loss < best_fold_loss:
                best_fold_loss = fold_loss
                best_fold_model = self.model.state_dict()

            print(f"Fold {fold_idx + 1} loss: {fold_loss}")
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {str(e)}")
            continue

    print("Cross-validation completed")

    if best_fold_model is not None:
        # Load the best model from cross-validation
        self.model.load_state_dict(best_fold_model)
        print(f"Best fold validation loss: {best_fold_loss}")
    else:
        print("No successful folds completed. Check your data and model.")

    def run_fold(self, train_loader, val_loader, optimizer):
        # Train and validate for one fold
        best_val_loss = float('inf')
        for epoch in range(self.config.max_epochs):
            self.run_epoch(train_loader, optimizer, is_train=True)
            val_loss = self.run_epoch(val_loader, optimizer, is_train=False)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return best_val_loss

def run_epoch(self, loader, optimizer, is_train=True):
    model = self.model
    model.train(is_train)
    losses = []
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y, p, v) in pbar:
        try:
            # Move data to the correct device
            x = x.to(self.device)
            y = y.to(self.device)
            p = p.to(self.device)
            v = v.to(self.device)

            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y, p, v, tokenizer=self.train_dataset.itos)
                loss = loss.mean()  # In case of multi-GPU setup

            if is_train:
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            pbar.set_description(f"{'train' if is_train else 'val'} loss: {loss.item():.5f}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: ran out of memory in {'training' if is_train else 'validation'}, skipping batch")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                print(f"RuntimeError in {'training' if is_train else 'validation'} batch {it}: {str(e)}")
            continue
        except Exception as e:
            print(f"Error in {'training' if is_train else 'validation'} batch {it}: {str(e)}")
            continue

    return float(np.mean(losses)) if losses else float('inf')

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
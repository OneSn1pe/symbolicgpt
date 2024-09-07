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


    def cross_validate(self, num_folds=5, seed=42):
        print(f"Starting cross-validation with {num_folds} folds")
    
        total_size = len(self.train_dataset)
        print(f"Total dataset size: {total_size}")

        folds = create_k_folds(self.train_dataset, num_folds=num_folds)
        print(f"Number of folds created: {len(folds)}")

        best_fold_loss = float('inf')
        best_fold_model = None

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            print(f"\nStarting fold {fold_idx + 1}/{num_folds}")
            print(f"Number of train indices: {len(train_indices)}")
            print(f"Number of val indices: {len(val_indices)}")
            print(f"Max train index: {max(train_indices)}, Min train index: {min(train_indices)}")
            print(f"Max val index: {max(val_indices)}, Min val index: {min(val_indices)}")

            if max(train_indices) >= total_size or max(val_indices) >= total_size:
                print(f"WARNING: Index out of bounds in fold {fold_idx + 1}")
                continue

            train_subset = torch.utils.data.Subset(self.train_dataset, train_indices)
            val_subset = torch.utils.data.Subset(self.train_dataset, val_indices)

            print(f"Train subset size: {len(train_subset)}")
            print(f"Val subset size: {len(val_subset)}")

            train_loader = DataLoader(train_subset, pin_memory=True,
                                  sampler=CPUSampler(train_subset),
                                  batch_size=self.config.batch_size,
                                  num_workers=self.config.num_workers)
            val_loader = DataLoader(val_subset, pin_memory=True,
                                sampler=CPUSampler(val_subset),
                                batch_size=self.config.batch_size,
                                num_workers=self.config.num_workers)

        # Reset the model
            self.model.apply(self.model.module._init_weights)
        
        # Initialize optimizer
            optimizer = self.model.module.configure_optimizers(self.config)
        
        # Run training and validation for the current fold
            try:
                fold_loss = self.run_fold(train_loader, val_loader, optimizer)
                print(f"Fold {fold_idx + 1} loss: {fold_loss}")

                if fold_loss < best_fold_loss:
                    best_fold_loss = fold_loss
                    best_fold_model = self.model.state_dict()
            except Exception as e:
                print(f"Error in fold {fold_idx + 1}: {str(e)}")
                continue

    # Load the best model from cross-validation
        if best_fold_model is not None:
            self.model.load_state_state(best_fold_model)
            print(f"Best fold validation loss: {best_fold_loss}")
        else:
            print("Warning: No best model found. Check if all folds failed.")



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
            x, y, p, v = x.to(self.device), y.to(self.device), p.to(self.device), v.to(self.device)
            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y, p, v, tokenizer=self.train_dataset.itos)
                loss = loss.mean()
                losses.append(loss.item())

            if is_train:
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_norm_clip)
                optimizer.step()

            pbar.set_description(f"{'train' if is_train else 'val'} loss {loss.item():.5f}")

        return float(np.mean(losses))

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
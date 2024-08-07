"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset

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

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, best=None, device='gpu', n_splits=5):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = 'cpu'
        self.n_splits = n_splits
        
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print('We are using the gpu now! device={}'.format(self.device))
        
        self.best_loss = best

    def cross_validate(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_dataset)):
            print(f'Fold {fold + 1}/{self.n_splits}')
            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)
            
            self.train(train_subset, val_subset)

            # Store the results
            fold_results.append(self.best_loss)
        
        avg_loss = np.mean(fold_results)
        print(f'Average validation loss across {self.n_splits} folds: {avg_loss:.4f}')
        return avg_loss

    def train(self, train_data, val_data):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(data, split):
            is_train = split == 'train'
            model.train(is_train)
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size, num_workers=config.num_workers)
            
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, p, v) in pbar:
                x, y, p, v = x.to(self.device), y.to(self.device), p.to(self.device), v.to(self.device)
                
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, p, v, tokenizer=self.train_dataset.itos)
                    loss = loss.mean()
                    losses.append(loss.item())
                
                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        self.best_loss = float('inf') if self.best_loss is None else self.best_loss
        self.tokens = 0

        for epoch in range(config.max_epochs):
            run_epoch(train_data, 'train')
            test_loss = run_epoch(val_data, 'val')
            
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                if self.config.ckpt_path is not None:
                    self.save_checkpoint()

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)



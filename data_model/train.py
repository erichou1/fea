# train.py - Training Script
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from models import FloorPlanTo3DPipeline
from dataset import create_dataloader
import numpy as np

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = FloorPlanTo3DPipeline(num_vertices=config['num_vertices'])
        self.model = self.model.to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )

        # Loss functions
        self.chamfer_loss = ChamferDistance()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        # Data loaders
        self.train_loader = create_dataloader(
            config['train_data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

        if config.get('val_data_dir'):
            self.val_loader = create_dataloader(
                config['val_data_dir'],
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                shuffle=False
            )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            floor_plans = [fp.to(self.device) for fp in batch['floor_plans']]
            exteriors = [ext.to(self.device) for ext in batch['exteriors']]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(floor_plans, exteriors)

            # Calculate losses
            # Note: You need ground truth 3D models for supervised training
            # Here we use self-supervised losses as example

            # 1. Vertex regularization (prevent degenerate solutions)
            vertex_reg_loss = torch.mean(torch.abs(outputs['vertices']))

            # 2. Material consistency loss
            material_var = torch.var(outputs['materials'], dim=1)
            material_loss = torch.mean(material_var)

            # 3. Structural soundness loss (encourage sound structures)
            soundness_target = torch.ones_like(outputs['is_structurally_sound'])
            soundness_loss = self.bce_loss(outputs['is_structurally_sound'], soundness_target)

            # 4. If you have ground truth vertices (chamfer distance)
            # loss_chamfer = self.chamfer_loss(outputs['vertices'], gt_vertices)

            # Combined loss
            loss = (
                0.1 * vertex_reg_loss +
                0.3 * material_loss +
                1.0 * soundness_loss
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train', loss.item(), global_step)

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                floor_plans = [fp.to(self.device) for fp in batch['floor_plans']]
                exteriors = [ext.to(self.device) for ext in batch['exteriors']]

                outputs = self.model(floor_plans, exteriors)

                # Calculate validation metrics
                soundness_acc = (outputs['is_structurally_sound'] > 0.5).float().mean()

                self.writer.add_scalar('Metrics/soundness_accuracy', soundness_acc, epoch)

    def save_checkpoint(self, epoch, filepath):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
        print(f"Checkpoint saved: {filepath}")

    def train(self):
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validate
            self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(
                    epoch,
                    os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
                )

        self.writer.close()

class ChamferDistance(nn.Module):
    """Chamfer Distance for comparing point clouds"""
    def forward(self, pred, target):
        # pred: [B, N, 3], target: [B, M, 3]
        pred = pred.unsqueeze(2)  # [B, N, 1, 3]
        target = target.unsqueeze(1)  # [B, 1, M, 3]

        dist = torch.sum((pred - target) ** 2, dim=-1)  # [B, N, M]

        dist_pred_to_target = torch.min(dist, dim=2)[0]  # [B, N]
        dist_target_to_pred = torch.min(dist, dim=1)[0]  # [B, M]

        return torch.mean(dist_pred_to_target) + torch.mean(dist_target_to_pred)

if __name__ == '__main__':
    config = {
        'num_vertices': 2048,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 8,
        'num_epochs': 100,
        'num_workers': 4,
        'train_data_dir': './data/train',
        'val_data_dir': './data/val',
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'save_every': 5
    }

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    trainer = Trainer(config)
    trainer.train()

# train_direct.py - Train directly from floor_plans2 and house_images2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from models import FloorPlanTo3DPipeline
from custom_dataset import create_direct_dataloader

class DirectTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = FloorPlanTo3DPipeline(num_vertices=config['num_vertices'])
        self.model = self.model.to(self.device)

        # Multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )

        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        # Data loader - DIRECT from your folders
        self.train_loader = create_direct_dataloader(
            floor_plans_dir=config['floor_plans_dir'],
            house_images_dir=config['house_images_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            shuffle=True
        )

        print(f"Training on {len(self.train_loader.dataset)} houses")

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

            # Loss calculation
            # 1. Vertex smoothness (encourage non-degenerate geometry)
            vertices = outputs['vertices']
            vertex_std = torch.std(vertices, dim=1).mean()
            vertex_loss = torch.abs(vertex_std - 3.0)  # Target std of ~3 meters

            # 2. Material consistency
            materials = outputs['materials']
            material_var = torch.var(materials, dim=1).mean()
            material_loss = material_var * 0.1

            # 3. Structural soundness (encourage sound structures)
            soundness_target = torch.ones_like(outputs['is_structurally_sound']) * 0.8
            soundness_loss = self.mse_loss(outputs['is_structurally_sound'], soundness_target)

            # 4. Structural properties regularization
            struct_props = outputs['structural_properties']
            # Encourage reasonable values
            stress_reg = torch.abs(struct_props[:, 0] - 50.0).mean()  # Target ~50 MPa
            safety_reg = torch.abs(struct_props[:, 2] - 2.5).mean()   # Target safety factor ~2.5

            # Combined loss
            loss = (
                0.3 * vertex_loss +
                0.2 * material_loss +
                1.0 * soundness_loss +
                0.1 * stress_reg +
                0.1 * safety_reg
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sound': f'{outputs["is_structurally_sound"].mean().item():.2f}'
            })

            # Log
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/total', loss.item(), global_step)
            self.writer.add_scalar('Loss/vertex', vertex_loss.item(), global_step)
            self.writer.add_scalar('Loss/soundness', soundness_loss.item(), global_step)
            self.writer.add_scalar('Metrics/soundness_prob', 
                                 outputs['is_structurally_sound'].mean().item(), 
                                 global_step)

        return total_loss / len(self.train_loader)

    def save_checkpoint(self, epoch, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved: {filepath}")

    def train(self):
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch}: Loss = {train_loss:.4f}, LR = {self.scheduler.get_last_lr()[0]:.6f}")

            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(
                    epoch,
                    os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
                )

        # Save final model
        self.save_checkpoint(
            self.config['num_epochs'] - 1,
            os.path.join(self.config['checkpoint_dir'], 'final_model.pth')
        )

        self.writer.close()
        print("\n✓ Training complete!")

if __name__ == '__main__':
    config = {
        # Data paths - YOUR FOLDERS
        'floor_plans_dir': 'floor_plans2',
        'house_images_dir': 'house_images2',

        # Model config
        'num_vertices': 2048,

        # Training config
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 4,  # Reduce if GPU memory issues
        'num_epochs': 50,
        'num_workers': 4,

        # Logging
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'save_every': 5
    }

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    trainer = DirectTrainer(config)
    trainer.train()

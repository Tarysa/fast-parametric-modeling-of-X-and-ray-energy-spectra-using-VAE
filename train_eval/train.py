import sys
import copy
import argparse
import time
import os
import json
from tqdm import tqdm
from natsort import natsorted

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

sys.path.append('../')
from utils.preprocessing import *
from models.model import *


def setup_argument_parser():
    """
    Set up command line argument parser with all training configuration options.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use_lr_scheduler', type=bool, default=False, help='Use learning rate scheduler')
    
    # Model architecture parameters
    parser.add_argument('--type', type=str, default="instancenorm", help='Normalization type')
    parser.add_argument('--nb_conv_layer', type=int, default=3, help='Number of convolutional layers')
    parser.add_argument('--nb_dense_layer', type=int, default=1, help='Number of dense layers')
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension size')
    parser.add_argument('--dropout', type=bool, default=False, help='Use dropout')
    parser.add_argument('--dropout_cls', type=bool, default=False, help='Use dropout in classifier')
    parser.add_argument('--resnet', type=bool, default=False, help='Use ResNet architecture')
    parser.add_argument('--film_layer', type=bool, default=False, help='Use FiLM layers')
    parser.add_argument('--model_version', type=int, default=1, help='Model version')
    
    # Loss function parameters
    parser.add_argument('--beta', type=float, default=1, help='Beta parameter for KL divergence')
    parser.add_argument('--last_act', type=str, default="identity", help='Last activation function')
    parser.add_argument('--loss', type=str, default="mse", help='Loss function type')
    parser.add_argument('--reduction', type=str, default="sum", help='Loss reduction method')
    parser.add_argument('--beta_schedule_ratio', type=float, default=0.1, help='Beta scheduling ratio')
    parser.add_argument('--alpha', type=float, default=0.3, help='Alpha parameter for domain adaptation')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=int, default=1, help='Dataset version to use')
    parser.add_argument('--version', type=int, default=3, help='Dataset processing version')
    parser.add_argument('--reduced_dataset', type=bool, default=False, help='Use reduced dataset')
    parser.add_argument('--norm_max', type=bool, default=False, help='Use max normalization')
    
    # Optimizer parameters
    parser.add_argument('--opt', type=str, default="sgd", help='Optimizer type (sgd, adam, adamw)')
    
    # Transfer learning parameters
    parser.add_argument('--transfer', type=bool, default=False, help='Use transfer learning')
    parser.add_argument('--date_pretrained', type=str, default=None, help='Date of pretrained model')
    parser.add_argument('--time_pretrained', type=str, default=None, help='Time of pretrained model')
    
    # Regression parameters
    parser.add_argument('--train_with_regressor', type=bool, default=False, help='Train with regression head')
    parser.add_argument('--beta_regressor', type=float, default=1, help='Beta parameter for regression loss')
    parser.add_argument('--regression_with_mu', type=bool, default=False, help='Use mu for regression')
    parser.add_argument('--train_with_regressor_with_only_mu', type=bool, default=False, 
                       help='Train regressor with only mu')
    
    # Logging parameters
    parser.add_argument('--date', type=str, default=time.strftime("%y%m%d"), help='Training date')
    parser.add_argument('--time', type=str, default=time.strftime("%H%M%S"), help='Training time')
    
    return parser


def setup_datasets(config):
    """
    Set up training and validation datasets based on configuration.
    
    Args:
        config: Configuration object with dataset parameters
        
    Returns:
        tuple: (train_dataset, val_dataset, inputs_dim, inputs_class)
    """
    if config.dataset == 1:
        # Dataset 1: Standard spectral data
        inputs_dim = 16384
        inputs_class = 3
        
        train_dataset = Dataset_dataset1('../datasets/MC/train.json', config)
        val_dataset = Dataset_dataset1('../datasets/MC/val.json', config)
        
    elif config.dataset == 3:
        # Dataset 3: Uranium spectral data with reduced dimensions
        inputs_dim = 4096
        
        train_dataset = Dataset_dataset3('../datasets/ESARDA/dataset U/train.json', 
                                       config)
        val_dataset = Dataset_dataset3('../datasets/ESARDA/dataset U/val.json', 
                                     config)
        inputs_class = train_dataset.inputs_class
        
    else:
        sys.exit(f"Unknown dataset for training: {config.dataset}")
    
    print(f"Dataset {config.dataset} loaded successfully")
    print(f"Input dimensions: {inputs_dim}")
    print(f"Number of physical parameters: {inputs_class}")
    
    return train_dataset, val_dataset, inputs_dim, inputs_class


def setup_data_loaders(train_dataset, val_dataset, config):
    """
    Set up PyTorch data loaders for training and validation.
    
    Args:
        train_dataset, val_dataset: Dataset objects
        config: Configuration object
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers, 
        shuffle=True
    )
    
    val_loader = data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader


def setup_optimizer(model, config):
    """
    Set up optimizer based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration object
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    if config.opt == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif config.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            momentum=0.9, 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.opt}")
    
    return optimizer


def setup_lr_scheduler(optimizer, config, train_dataset):
    """
    Set up learning rate scheduler if enabled.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration object
        train_dataset: Training dataset
        
    Returns:
        torch.optim.lr_scheduler or None: Learning rate scheduler
    """
    if not config.use_lr_scheduler:
        return None
    
    dataset_size = len(train_dataset)
    total_steps = (1 - config.beta_schedule_ratio) * config.n_epochs * (dataset_size // config.batch_size)
    print(f"Total steps for OneCycleLR scheduler: {total_steps}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2e-4, 
        total_steps=total_steps, 
        final_div_factor=10
    )
    
    return scheduler


def load_pretrained_model(model, optimizer, config):
    """
    Load pretrained model weights if transfer learning is enabled.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer  
        config: Configuration object
        
    Returns:
        float: Starting beta value for transfer learning
    """
    if not config.transfer:
        return None
    
    beta_start = config.beta
    
    # Get pretrained model path from user input if not provided
    if config.date_pretrained is None and config.time_pretrained is None:
        config.date_pretrained = input("Date of pretrained model: ")
        config.time_pretrained = input("Time of pretrained model: ")
    
    # Load pretrained weights
    load_path = f"../Results/VAE/{config.date_pretrained}/{config.time_pretrained}"
    filename = natsorted([
        file for file in os.listdir(load_path) 
        if not file.endswith(".txt")
    ])[-1]
    
    weight = torch.load(os.path.join(load_path, filename))
    
    model.load_state_dict(weight["weights"])
    optimizer.load_state_dict(weight["opt_vae"])
    
    print(f"Loaded pretrained model from: {load_path}/{filename}")
    
    return beta_start


def setup_loss_functions(config, beta_start=None):
    """
    Set up loss functions based on configuration.
    
    Args:
        config: Configuration object
        beta_start: Starting beta value for transfer learning
        
    Returns:
        tuple: (vae_loss, regression_loss)
    """
    # Set up VAE loss with beta scheduling
    if beta_start is not None:
        # Transfer learning case
        vae_loss = VAELossWithSchedule(
            config.n_epochs, 
            start=beta_start, 
            stop=config.beta, 
            ratio=config.beta_schedule_ratio,
            loss=config.loss, 
        )
    else:
        # Regular training case
        vae_loss = VAELossWithSchedule(
            config.n_epochs, 
            stop=config.beta, 
            ratio=config.beta_schedule_ratio,
            loss=config.loss, 
        )
    
    # Set up regression loss if needed
    regression_loss = None
    if config.train_with_regressor:
        if config.norm_max:
            regression_loss = torch.nn.L1Loss()
        else:
            regression_loss = torch.nn.MSELoss()
    
    return vae_loss, regression_loss


def train_epoch(model, train_loader, optimizer, vae_loss, regression_loss, config, epoch, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: PyTorch optimizer
        vae_loss: VAE loss function
        regression_loss: Regression loss function
        config: Configuration object
        epoch: Current epoch number
        device: Device to run on
        
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    
    # Initialize loss accumulators
    total_loss = 0
    kl_loss = 0
    mse_loss = 0
    reg_loss = 0
    domain_loss = 0
    
    dataset_size = len(train_loader.dataset)
    
    for batch, inputs in enumerate(train_loader):
        # Unpack inputs based on configuration
        if config.train_with_regressor:
            X, condition, norm_param = inputs
            norm_param = norm_param.to(device)
        else:
            X, condition = inputs
        
        X, condition = X.to(device), condition.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if config.train_with_regressor:
            output_VAE, pred_norm = model(X, condition)
            regloss = torch.sqrt(regression_loss(pred_norm, norm_param.float()))
        else:
            if config.model_version == 2:
                output_VAE, dann_loss = model(X, condition, type=train_loader.dataset.condition_type)
                domain_loss += dann_loss.item()
            else:
                output_VAE = model(X, condition)
        
        # Calculate VAE loss
        loss_total, mse_loss_batch, kl_loss_batch = vae_loss(epoch, output_VAE, X)
        
        # Add additional losses
        if config.model_version == 2:
            loss_total += dann_loss
        elif config.train_with_regressor:
            loss_total += config.beta_regressor * regloss
            reg_loss += regloss.item()
        
        # Accumulate losses
        total_loss += loss_total.item()
        mse_loss += mse_loss_batch.item()
        kl_loss += kl_loss_batch.item()
        
        # Backward pass and optimization
        loss_total.backward()
        optimizer.step()
    
    # Average losses over dataset
    return {
        'total_loss': total_loss / dataset_size,
        'kl_loss': kl_loss / dataset_size,
        'mse_loss': mse_loss / dataset_size,
        'reg_loss': reg_loss / dataset_size,
        'domain_loss': domain_loss / dataset_size
    }


def validate_epoch(model, val_loader, vae_loss, regression_loss, config, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        vae_loss: VAE loss function
        regression_loss: Regression loss function
        config: Configuration object
        device: Device to run on
        
    Returns:
        dict: Validation metrics for the epoch
    """
    model.eval()
    
    # Initialize loss accumulators
    total_loss = 0
    kl_loss = 0
    mse_loss = 0
    reg_loss = 0
    domain_loss = 0
    
    dataset_size = len(val_loader.dataset)
    
    with torch.no_grad():
        for batch, inputs in enumerate(val_loader):
            # Unpack inputs based on configuration
            if config.train_with_regressor:
                X, condition, norm_param = inputs
                norm_param = norm_param.to(device)
            else:
                X, condition = inputs
            
            X, condition = X.to(device), condition.to(device)
            
            # Forward pass
            if config.train_with_regressor:
                output_VAE, pred_norm = model(X, condition)
                regloss = torch.sqrt(regression_loss(pred_norm, norm_param.float()))
            else:
                if config.model_version == 2:
                    output_VAE, dann_loss = model(X, condition, type=val_loader.dataset.condition_type)
                    domain_loss += dann_loss.item()
                else:
                    output_VAE = model(X, condition)
            
            # Calculate VAE loss (use final epoch for consistent validation)
            loss_total, mse_loss_batch, kl_loss_batch = vae_loss(config.n_epochs - 1, output_VAE, X)
            
            # Add additional losses
            if config.model_version == 2:
                loss_total += dann_loss
            elif config.train_with_regressor:
                loss_total += config.beta_regressor * regloss
                reg_loss += regloss.item()
            
            # Accumulate losses
            total_loss += loss_total.item()
            mse_loss += mse_loss_batch.item()
            kl_loss += kl_loss_batch.item()
    
    # Average losses over dataset
    return {
        'total_loss': total_loss / dataset_size,
        'kl_loss': kl_loss / dataset_size,
        'mse_loss': mse_loss / dataset_size,
        'reg_loss': reg_loss / dataset_size,
        'domain_loss': domain_loss / dataset_size
    }


def save_model_checkpoint(model, optimizer, config, epoch, best_epoch, is_best=False, is_final=False):
    """
    Save model checkpoint to disk.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        config: Configuration object
        epoch: Current epoch
        best_epoch: Best epoch number
        is_best: Whether this is the best model
        is_final: Whether this is the final model
    """
    # Create save directory
    save_path = f'../weights/{config.date}'
    os.makedirs(save_path, exist_ok=True)
    
    save_path = f'{save_path}/{config.time}'
    os.makedirs(save_path, exist_ok=True)
    
    # Update config with current information
    config.last_epoch = epoch
    config.best_epoch = best_epoch
    
    # Save configuration
    with open(f'{save_path}/config.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Save model weights
    checkpoint = {
        "weights": model.state_dict(),
        "opt_vae": optimizer.state_dict()
    }
    
    if is_best:
        torch.save(checkpoint, f'{save_path}/best_model.pt')
    
    if is_final or (epoch % 1000 == 0):
        torch.save(checkpoint, f'{save_path}/last_model.pt')


def main():
    """Main training loop for VAE model."""
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # =============================================================================
    # 1. SETUP AND CONFIGURATION
    # =============================================================================
    
    # Parse command line arguments
    parser = setup_argument_parser()
    config = parser.parse_args()
    
    # Set up TensorBoard logging
    run_name = (f'{config.date}/{config.time}')
    writer = SummaryWriter(f'../runs/{run_name}')
    
    print('=' * 50)
    print('Training Configuration:')
    print(config)
    print('=' * 50)
    
    # =============================================================================
    # 2. DATA SETUP
    # =============================================================================
    
    # Set up datasets
    train_dataset, val_dataset, inputs_dim, inputs_class = setup_datasets(config)
    config.inputs_class = inputs_class
    
    # Set up data loaders
    train_loader, val_loader = setup_data_loaders(train_dataset, val_dataset, config)
    
    # =============================================================================
    # 3. MODEL SETUP
    # =============================================================================
    
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Initialize model
    model = VAE(config, device, inputs_dim, inputs_class).to(device)
    
    # Print model summary
    if config.model_version == 2:
        print(summary(model, [(inputs_dim,), (inputs_class,), (inputs_class,)]))
    else:
        print(summary(model, [(inputs_dim,), (inputs_class,)]))
    
    # Set up optimizer
    optimizer = setup_optimizer(model, config)
    
    # Set up learning rate scheduler
    scheduler = setup_lr_scheduler(optimizer, config, train_dataset)
    
    # Load pretrained model if transfer learning
    beta_start = load_pretrained_model(model, optimizer, config)
    
    # Set up loss functions
    vae_loss, regression_loss = setup_loss_functions(config, beta_start)
    
    # =============================================================================
    # 4. TRAINING LOOP
    # =============================================================================
    
    # Initialize best model tracking
    best_loss = float('inf')
    best_model = None
    best_optimizer = None
    best_epoch = 0
    
    print("Starting training...")
    
    for epoch in tqdm(range(config.n_epochs), desc="Training Progress"):
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, vae_loss, regression_loss, 
            config, epoch, device
        )
        
        # Validate for one epoch
        val_metrics = validate_epoch(
            model, val_loader, vae_loss, regression_loss, config, device
        )
        
        # Update best model if validation loss improved
        if val_metrics['total_loss'] <= best_loss:
            best_loss = val_metrics['total_loss']
            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)
            best_epoch = epoch
        
        # Log metrics to TensorBoard
        writer.add_scalar("loss/training", train_metrics['total_loss'], epoch)
        writer.add_scalar("loss/validation", val_metrics['total_loss'], epoch)
        writer.add_scalar("Reconstruction Loss/training", train_metrics['mse_loss'], epoch)
        writer.add_scalar("Reconstruction Loss/validation", val_metrics['mse_loss'], epoch)
        writer.add_scalar("KL loss/training", train_metrics['kl_loss'], epoch)
        writer.add_scalar("KL loss/validation", val_metrics['kl_loss'], epoch)
        writer.add_scalar("Regression Loss/training", train_metrics['reg_loss'], epoch)
        writer.add_scalar("Regression Loss/validation", val_metrics['reg_loss'], epoch)
        writer.add_scalar("DANN loss/training", train_metrics['domain_loss'], epoch)
        writer.add_scalar("DANN Loss/validation", val_metrics['domain_loss'], epoch)
        
        # Save checkpoints periodically and at the end
        if (epoch % 100 == 0) or (epoch == config.n_epochs - 1):
            # Save best model
            save_model_checkpoint(
                best_model, best_optimizer, config, epoch, best_epoch, is_best=True
            )
            # Save current model
            save_model_checkpoint(
                model, optimizer, config, epoch, best_epoch, is_final=(epoch == config.n_epochs - 1)
            )
    
    # =============================================================================
    # 5. CLEANUP
    # =============================================================================
    
    writer.close()
    torch.cuda.empty_cache()
    
    print('=' * 50)
    print('Training completed successfully!')
    print(f'Best epoch: {best_epoch}')
    print(f'Best validation loss: {best_loss:.6f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
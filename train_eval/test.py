"""
VAE Model Evaluation Script

This script evaluates a trained Variational Autoencoder (VAE) model with regression capabilities
for signal reconstruction and parameter estimation. It performs comprehensive analysis including:
- Model loading and evaluation
- Signal reconstruction quality assessment
- Latent space visualization
- Parameter regression performance evaluation
- Results export to CSV files
"""

import sys
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from tqdm import tqdm
from torch.utils import data
from torchsummary import summary
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error
from torchview import draw_graph

sys.path.append('../')
# Custom imports
from utils.functions import reconstruction_signal, latent_space_visualisation
from utils.preprocessing import Dataset_dataset1, Dataset_dataset3
from models.model import VAE, Regressor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, type=int, help='Dataset used')
    parser.add_argument('--draw_architecture', default=False, type=bool, help='Draw model architecture')
    parser.add_argument('--make_csv_file', default=False, type=bool, help='Generate CSV output files')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config, _ = parser.parse_known_args()
    
    with open(f'{config_path}/config.txt', 'r') as f:
        config.__dict__ = json.load(f)
    
    return config


def set_default_config_values(config):
    """Set default values for missing configuration parameters."""
    defaults = {
        'last_act': "identity",
        'regression_with_mu': False,
        'train_with_regressor_with_only_mu': False,
        'model_version': 1
    }
    
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    # Version-specific defaults
    if config.model_version == 2:
        config.norm_max = True


def load_datasets(config):
    """Load appropriate datasets based on configuration."""
    if config.dataset == 1:
        inputs_dim = 16384
        inputs_class = 3
        
        nb_classes = 1
        train_dataset = Dataset_dataset1('../datasets/MC/train.json', config, not_training=True)
        val_dataset = Dataset_dataset1('../datasets/MC/val.json', config, not_training=True)
        test_dataset = Dataset_dataset1('../datasets/MC/test.json', config, not_training=True)
    
        # Load pre-normalization datasets for evaluation
        train_dataset_before_norm = Dataset_dataset1('../datasets/MC/train_before_normalization.json', config, before_norm=True)
        val_dataset_before_norm = Dataset_dataset1('../datasets/MC/val_before_normalization.json', config, before_norm=True)
        test_dataset_before_norm = Dataset_dataset1('../datasets/MC/test_before_normalization.json', config, before_norm=True)
        
    elif config.dataset == 3:
        inputs_dim = 4096
        nb_classes = 1
        
        train_dataset = Dataset_dataset3('../datasets/ESARDA/dataset U/train.json', config, not_training=True)
        val_dataset = Dataset_dataset3('../datasets/ESARDA/dataset U/val.json', config, not_training=True)
        test_dataset = Dataset_dataset3('../datasets/ESARDA/dataset U/test.json', config, not_training=True)

        # Load pre-normalization datasets for evaluation
        train_dataset_before_norm = Dataset_dataset3('../datasets/ESARDA/dataset U/train_before_normalization_reduced.json', config, before_norm=True)
        val_dataset_before_norm = Dataset_dataset3('../datasets/ESARDA/dataset U/val_before_normalization_reduced.json', config, before_norm=True)
        test_dataset_before_norm = Dataset_dataset3('../datasets/ESARDA/dataset U/test_before_normalization_reduced.json', config, before_norm=True)
    
        inputs_class = train_dataset.inputs_class
    
    return (train_dataset, val_dataset, test_dataset, 
            train_dataset_before_norm, val_dataset_before_norm, test_dataset_before_norm,
            inputs_dim, inputs_class, nb_classes)


def load_models(config, load_path_vae, inputs_dim, inputs_class, device):
    """Load VAE and regressor models."""
    # Load VAE model
    model = VAE(config, device, inputs_dim, inputs_class).to(device)
    
    filename = "last_model.pt"
    weights = torch.load(load_path_vae + "/" + filename)
    
    if isinstance(weights, dict):
        model.load_state_dict(weights["weights"])
    else:
        model.load_state_dict(weights)
    
    model.eval()
    
    # Load pre-trained sklearn regressor
    regressor = joblib.load(f"../weights/regression_max_dataset{config.dataset}.joblib")
       
    return model, regressor


def evaluate_model(model, regressor, data_loader, config, device, dataset_name=""):
    """Evaluate model on given dataset."""
    truth_data = []
    reconstruction_data = []
    condition_data = []
    mu_data = []
    sigma_data = []
    norm_param_pred_data = []
    norm_param_truth_data = []
    error_reconstruction = []
    
    with torch.no_grad():
        for batch, (X, condition, norm_param) in enumerate(data_loader):
            X, condition, norm_param = X.to(device), condition.to(device), norm_param.to(device)
            
            # Forward pass through VAE
            if config.train_with_regressor:
                reconstruction, mu, sigma, norm_param_pred = model.exact_reconstruction(X, condition)
            else:
                # Handle different dataset conditions
                if config.dataset == 2:
                    reconstruction, mu, sigma = model.exact_reconstruction(X, condition[:, :-2])
                else:
                    reconstruction, mu, sigma = model.exact_reconstruction(X, condition)
                
                # Predict normalization parameters
                if config.model_version == 1:
                    norm_param_pred = regressor(mu).cpu().detach().numpy()
                else:
                    norm_param_pred = regressor.predict(condition.cpu().numpy()).reshape(-1, 1)
            
            # Store results
            truth_data.append(X.cpu().detach().numpy())
            reconstruction_data.append(reconstruction.cpu().detach().numpy())
            condition_data.append(condition.cpu().detach().numpy())
            mu_data.append(mu.cpu().detach().numpy())
            sigma_data.append(torch.exp(0.5 * sigma).cpu().detach().numpy())
            norm_param_truth_data.append(norm_param.cpu().detach().numpy())
            norm_param_pred_data.append(norm_param_pred)
            
            # Calculate reconstruction error (MSE)
            reconstruction_error = ((X.cpu().detach().numpy() - reconstruction.cpu().detach().numpy()) ** 2).mean(axis=1)
            error_reconstruction.append(reconstruction_error)
    
    # Concatenate all batches
    results = {
        'truth': np.concatenate(truth_data, axis=0),
        'reconstruction': np.concatenate(reconstruction_data, axis=0),
        'condition': np.concatenate(condition_data, axis=0),
        'mu': np.concatenate(mu_data, axis=0),
        'sigma': np.concatenate(sigma_data, axis=0),
        'norm_param_truth': np.concatenate(norm_param_truth_data, axis=0),
        'norm_param_pred': np.concatenate(norm_param_pred_data, axis=0),
        'error_reconstruction': np.concatenate(error_reconstruction, axis=0)
    }
    
    return results


def calculate_metrics(results_train, results_val, results_test, train_dataset, 
                     train_before_norm, val_before_norm, test_before_norm):
    """Calculate evaluation metrics."""
    # Denormalize reconstructions
    reconstruction_train_denorm = train_dataset.undo_normalization(
        results_train['reconstruction'], results_train['norm_param_pred']
    )
    reconstruction_val_denorm = train_dataset.undo_normalization(
        results_val['reconstruction'], results_val['norm_param_pred']
    )
    reconstruction_test_denorm = train_dataset.undo_normalization(
        results_test['reconstruction'], results_test['norm_param_pred']
    )
    
    # Calculate MAE on denormalized data
    mae_train = np.abs(train_before_norm - reconstruction_train_denorm).mean(axis=1)
    mae_val = np.abs(val_before_norm - reconstruction_val_denorm).mean(axis=1)
    mae_test = np.abs(test_before_norm - reconstruction_test_denorm).mean(axis=1)
    
    # Calculate regression errors
    reg_error_train = np.abs(results_train['norm_param_truth'] - results_train['norm_param_pred'])
    reg_error_val = np.abs(results_val['norm_param_truth'] - results_val['norm_param_pred'])
    reg_error_test = np.abs(results_test['norm_param_truth'] - results_test['norm_param_pred'])
    
    return {
        'reconstruction_denorm': {
            'train': reconstruction_train_denorm,
            'val': reconstruction_val_denorm,
            'test': reconstruction_test_denorm
        },
        'mae': {
            'train': mae_train,
            'val': mae_val,
            'test': mae_test
        },
        'regression_error': {
            'train': reg_error_train,
            'val': reg_error_val,
            'test': reg_error_test
        }
    }


def create_visualizations(results_train, results_val, results_test, config, path, detector_name = None):
    """Create various visualizations for model evaluation."""
    # Combine all results
    all_mu = np.concatenate([results_train['mu'], results_val['mu'], results_test['mu']], axis=0)
    all_condition = np.concatenate([results_train['condition'], results_val['condition'], results_test['condition']], axis=0)
    all_norm_param = np.concatenate([
        results_train['norm_param_truth'], 
        results_val['norm_param_truth'], 
        results_test['norm_param_truth']
    ])
    
    # Create t-SNE visualization of latent space
    print("Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    latent_space_2d = tsne.fit_transform(all_mu)
    
    # Visualize latent space with different conditions
    if config.dataset == 1:
        condition_names = ["type", "enrichment", "attenuation"]
        condition_types = ["quali", "quanti", "quanti"]
        
        for n_condition, (condition_name, condition_type) in enumerate(zip(condition_names, condition_types)):

            if condition_name == "type" and detector_name is not None:
                labels_classe = detector_name
            else:
                labels_classe = np.unique(all_condition[:,n_condition])

            latent_space_visualisation(
                latent_space_2d, 
                all_condition[:, n_condition], 
                labels_classe, 
                f"{path}/LS_visualisation_hue_{condition_name}", 
                viridis=True, 
                type=condition_type
            )
    else:
        condition_names = ['Detector quanti', 'Declared enrichment scaled', 'real counting times scaled']
        condition_types = ["quali", "quanti", "quanti"]
        labels_classe = []
        for n_condition, (condition_name, condition_type) in enumerate(zip(condition_names, condition_types)):

            if condition_name == "Detector quanti" and detector_name is not None:
                labels_classe = detector_name
            else:
                labels_classe = np.unique(all_condition[:,n_condition])
            latent_space_visualisation(latent_space_2d, all_condition[:,n_condition], labels_classe, f"{path}/LS visualisation hue {condition_name}", viridis = True, type = condition_type)
    
    # Visualize by normalization parameter
    latent_space_visualisation(
        latent_space_2d, 
        all_norm_param, 
        np.unique(all_norm_param), 
        f"{path}/LS_visualisation_hue_max_amp", 
        viridis=True, 
        type="quanti"
    )
    
    # Visualize by dataset split
    dataset_labels = np.array([0] * len(results_train['mu']) + 
                             [1] * len(results_val['mu']) + 
                             [2] * len(results_test['mu']))
    latent_space_visualisation(
        latent_space_2d, 
        dataset_labels, 
        ["train", "val", "test"], 
        f"{path}/LS_visualisation_train_val_test", 
        type="quali"
    )
    
    return latent_space_2d


def save_results(results_train, results_val, results_test, metrics, config, path, make_csv):
    """Save evaluation results and statistics."""
    if not make_csv:
        return
    
    # Create comprehensive results DataFrame
    # Implementation depends on specific requirements
    print("Saving results to CSV files...")
    
    # Save latent space coordinates with t-SNE
    # Implementation details...


def plot_r2_scores(results_train, results_val, results_test, config, path):
    """Plot R² scores for parameter prediction."""
    if config.version == 4:  # Skip for version 4
        return
    
    datasets = [
        (results_train, "train"),
        (results_val, "val"), 
        (results_test, "test")
    ]
    
    for results, dataset_name in datasets:
        fig = plt.figure(figsize=(8, 6))
        
        if config.version == 2025:
            # Handle multi-output case
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            r2_mean = r2_score(results['norm_param_truth'][:, 0], results['norm_param_pred'][:, 0])
            r2_std = r2_score(results['norm_param_truth'][:, 1], results['norm_param_pred'][:, 1])
            
            ax[0].scatter(results['norm_param_truth'][:, 0], results['norm_param_pred'][:, 0])
            ax[0].plot([0, 1], [0, 1], color='r')
            ax[0].set_title(f"R² Mean: {r2_mean:.4f}")
            ax[0].set_xlabel("Truth")
            ax[0].set_ylabel("Prediction")
            
            ax[1].scatter(results['norm_param_truth'][:, 1], results['norm_param_pred'][:, 1])
            ax[1].plot([0, 1], [0, 1], color='r')
            ax[1].set_title(f"R² Std: {r2_std:.4f}")
            ax[1].set_xlabel("Truth")
            ax[1].set_ylabel("Prediction")
        else:
            # Single output case
            r2_score_val = r2_score(results['norm_param_truth'], results['norm_param_pred'])
            
            plt.scatter(results['norm_param_truth'], results['norm_param_pred'])
            plt.plot([0, 1], [0, 1], color='r')
            plt.xlabel("Truth")
            plt.ylabel("Prediction")
            plt.title(f"R²: {r2_score_val:.4f}")
        
        plt.tight_layout()
        fig.savefig(f"{path}/R2_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function."""
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Parse arguments
    config_test = parse_arguments()
    
    print('=' * 50)
    print(f'Starting VAE Model Evaluation with dataset {config_test.dataset}')
    print('=' * 50)
    
    if config_test.dataset == 1:
        dataset_name = "MC"
    else:
        dataset_name = "ESARDA"
    

    # Load VAE configuration
    load_path_vae = f"../weights/{dataset_name}"
    config = load_config(load_path_vae)
    set_default_config_values(config)
  
    # Load datasets
    (train_dataset, val_dataset, test_dataset, 
     train_before_norm, val_before_norm, test_before_norm,
     inputs_dim, inputs_class, nb_classes) = load_datasets(config)
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    
    # Load models
    model, regressor = load_models(config, load_path_vae, inputs_dim, inputs_class, device)
    
    # Print model summary
    print(f"Input class dimension: {inputs_class}")
    if config.model_version == 2:
        print(summary(model, [(inputs_dim,), (inputs_class,), (inputs_class,)]))
    else:
        print(summary(model, [(inputs_dim,), (inputs_class,)]))
    
    # Create output directory
    output_path = f"../outputs/{dataset_name}"
    os.makedirs(output_path, exist_ok=True)
    
    # Draw architecture if requested
    if config_test.draw_architecture:
        model_graph = draw_graph(
            model, 
            input_size=((4, inputs_dim), (4, inputs_class)), 
            save_graph=True, 
            filename=f"{output_path}/architecture", 
            expand_nested=True
        )
    
    print("Evaluating model...")
    
    # Evaluate model on all datasets
    results_train = evaluate_model(model, regressor, train_loader, config, device, "train")
    results_val = evaluate_model(model, regressor, val_loader, config, device, "val")
    results_test = evaluate_model(model, regressor, test_loader, config, device, "test")
    
    # Get before normalization data
    train_before_norm_data = np.array([elem for elem in train_before_norm.data.values])
    val_before_norm_data = np.array([elem for elem in val_before_norm.data.values])
    test_before_norm_data = np.array([elem for elem in test_before_norm.data.values])
    
    # Calculate metrics
    metrics = calculate_metrics(
        results_train, results_val, results_test, train_dataset,
        train_before_norm_data, val_before_norm_data, test_before_norm_data
    )
    
    # Create visualizations
    if config_test.dataset == 1:
        detector_name = train_dataset.detector_name
    else:
        detector_name = train_dataset.detector_name
    latent_space_2d = create_visualizations(results_train, results_val, results_test, config, output_path, detector_name = detector_name)
    
    # Plot R² scores
    plot_r2_scores(results_train, results_val, results_test, config, output_path)
    
    # Plot some reconstruction examples
    print("Creating reconstruction plots...")
    random_indices_train = np.random.choice(len(results_train['reconstruction']), size=5, replace=False)
    random_indices_val = np.random.choice(len(results_val['reconstruction']), size=5, replace=False)
    random_indices_test = np.random.choice(len(results_test['reconstruction']), size=5, replace=False)

    reconstruction_signal(
        metrics['reconstruction_denorm']['train'][random_indices_train],
        train_before_norm_data[random_indices_train],
        f"{output_path}/reconstruction_plots_train_original_signal"
    )
    reconstruction_signal(
        metrics['reconstruction_denorm']['val'][random_indices_val],
        val_before_norm_data[random_indices_val],
        f"{output_path}/reconstruction_plots_val_original_signal"
    )

    reconstruction_signal(
        metrics['reconstruction_denorm']['test'][random_indices_test],
        test_before_norm_data[random_indices_test],
        f"{output_path}/reconstruction_plots_test_original_signal"
    )

    reconstruction_signal(
        results_train['reconstruction'][random_indices_train],
        results_train['truth'][random_indices_train],
        f"{output_path}/reconstruction_plots_train_normalized_signal"
    )

    reconstruction_signal(
        results_val['reconstruction'][random_indices_val],
        results_val['truth'][random_indices_val],
        f"{output_path}/reconstruction_plots_val_normalized_signal"
    )

    reconstruction_signal(
        results_test['reconstruction'][random_indices_test],
        results_test['truth'][random_indices_test],
        f"{output_path}/reconstruction_plots_test_normalized_signal"
    )
    
    
    # Save results
    save_results(results_train, results_val, results_test, metrics, config, output_path, config_test.make_csv_file)
    
    # Write summary statistics
    with open(f"{output_path}/evaluation_summary.txt", "w") as f:
        f.write("VAE Model Evaluation Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Dataset: {config.dataset}\n")
        f.write(f"Latent dimension: {config.latent_dim}\n")
        f.write(f"Model version: {config.model_version}\n\n")
        
        f.write("Reconstruction Errors (MAE on denormalized data):\n")
        f.write(f"  Train: {np.mean(metrics['mae']['train']):.6f}\n")
        f.write(f"  Val:   {np.mean(metrics['mae']['val']):.6f}\n")
        f.write(f"  Test:  {np.mean(metrics['mae']['test']):.6f}\n\n")
        
        f.write("Reconstruction Errors (MSE on normalized data):\n")
        f.write(f"  Train: {np.mean(results_train['error_reconstruction']):.6f}\n")
        f.write(f"  Val:   {np.mean(results_val['error_reconstruction']):.6f}\n")
        f.write(f"  Test:  {np.mean(results_test['error_reconstruction']):.6f}\n\n")
        
        f.write("Parameter Regression Errors:\n")
        f.write(f"  Train: {np.mean(metrics['regression_error']['train']):.6f}\n")
        f.write(f"  Val:   {np.mean(metrics['regression_error']['val']):.6f}\n")
        f.write(f"  Test:  {np.mean(metrics['regression_error']['test']):.6f}\n")
    
    print("=" * 50)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {output_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
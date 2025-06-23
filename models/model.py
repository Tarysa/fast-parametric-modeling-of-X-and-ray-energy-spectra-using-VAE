import torch
import torch.nn as nn
import numpy as np
from models.GRL import ReverseLayerF


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies affine transformation to features based on conditioning parameters.
    """
    def __init__(self, input_neurons, output_neurons):
        super().__init__()
        
        # Linear layers for gamma (scale) and beta (shift) parameters
        self.linear_gamma = nn.Linear(input_neurons, output_neurons)
        self.linear_beta = nn.Linear(input_neurons, output_neurons)
        
        # Initialize weights using Xavier uniform distribution
        nn.init.xavier_uniform_(self.linear_gamma.weight)
        nn.init.xavier_uniform_(self.linear_beta.weight)
    
    def forward(self, x, conditioning_params):
        """
        Apply FiLM transformation: x * gamma + beta
        
        Args:
            x: Input features [batch_size, features, sequence_length]
            conditioning_params: Conditioning parameters [batch_size, input_neurons]
        
        Returns:
            Modulated features
        """
        gamma = self.linear_gamma(conditioning_params).unsqueeze(-1)  # [batch, features, 1]
        beta = self.linear_beta(conditioning_params).unsqueeze(-1)    # [batch, features, 1]
        
        return x * gamma + beta


def compute_loss(predictions, targets, type, dataset_id=1):
    """
    Compute loss based on data types (qualitative or quantitative).
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        type: List of data types ('quali' or 'quanti')
        dataset_id: Dataset identifier (1 or 3)
    
    Returns:
        Combined loss value
    """
    total_loss = 0
    num_categories = 4  # Number of categorical classes
    
    for i, data_type in enumerate(type):
        if data_type == "quali":  # Qualitative (categorical) data
            if dataset_id == 1:
                total_loss += nn.BCEWithLogitsLoss()(predictions[:, i], targets[:, i])
            elif dataset_id == 3:
                total_loss += nn.CrossEntropyLoss()(
                    predictions[:, i:i+num_categories], 
                    targets[:, i].long()
                )
        else:  # Quantitative (continuous) data
            if dataset_id == 1:
                total_loss += nn.MSELoss()(predictions[:, i], targets[:, i])
            else:
                total_loss += nn.MSELoss()(
                    predictions[:, (i-1)+num_categories], 
                    targets[:, i]
                )
    
    return total_loss


class DenseBlock(nn.Module):
    """
    Dense (fully connected) block with optional normalization and activation.
    """
    def __init__(self, config, input_neurons, output_neurons, 
                 is_last_layer=False, use_norm=True, activation_function="leakyReLU"):
        super().__init__()
        self.config = config
        
        # Build activation sequence
        activation_layers = []
        
        if not is_last_layer:
            if use_norm:
                activation_layers.append(nn.BatchNorm1d(output_neurons))
                
                # Add specified activation function
                if activation_function == "leakyReLU":
                    activation_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                elif activation_function == "ReLU":
                    activation_layers.append(nn.ReLU())
                elif activation_function == "Mish":
                    activation_layers.append(nn.Mish())
                elif activation_function == "SiLU":
                    activation_layers.append(nn.SiLU())
        
        self.linear = nn.Linear(input_neurons, output_neurons)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.activation = nn.Sequential(*activation_layers)
    
    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)


class ConvBlock(nn.Module):
    """
    1D Convolutional block with optional normalization, activation, and dropout.
    """
    def __init__(self, config, input_channels, output_channels, kernel_size=4, 
                 use_activation=True, use_stride=True, use_norm=True, use_bias=True):
        super().__init__()
        self.config = config
        
        # Dropout configuration
        self.use_dropout = getattr(config, 'dropout', False)
        self.dropout_rate = getattr(config, 'drop_rate', 0.2)
        
        activation_layers = []
        
        if use_norm:
            # Choose normalization type
            if config.type == "batchnorm":
                activation_layers.append(nn.BatchNorm1d(output_channels))
            elif config.type == "instancenorm":
                activation_layers.append(nn.InstanceNorm1d(output_channels, affine=True))
            
            if use_activation:
                activation_layers.append(nn.ReLU(inplace=True))
        
        # Configure convolution layer
        if use_stride:
            self.conv = nn.Conv1d(input_channels, output_channels, 
                                kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)
        else:
            self.conv = nn.Conv1d(input_channels, output_channels, 
                                kernel_size=kernel_size, stride=1, padding="same", bias=use_bias)
        
        if self.use_dropout:
            activation_layers.append(nn.Dropout(p=self.dropout_rate))
        
        nn.init.xavier_uniform_(self.conv.weight)
        self.activation = nn.Sequential(*activation_layers)
    
    def forward(self, x, condition=None):
        x = self.conv(x)
        return self.activation(x)


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with optional FiLM conditioning.
    """
    def __init__(self, config, input_channels, output_channels, kernel_size, downsample):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsample = downsample
        self.use_film = getattr(config, 'film_layer', False)
        
        # Normalization layers
        norm_cls = (nn.BatchNorm1d if config.type == "batchnorm" 
                   else lambda c: nn.InstanceNorm1d(c, affine=True))
        self.norm1 = norm_cls(input_channels)
        self.norm2 = norm_cls(output_channels)
        
        self.activation = nn.ReLU()
        
        # FiLM layers for conditional generation
        if self.use_film:
            self.film_layer_1 = FiLMLayer(config.inputs_class, output_channels)
            self.film_layer_2 = FiLMLayer(config.inputs_class, output_channels)
        
        # Residual connection handling
        if downsample:
            self.downsample_conv1 = ConvBlock(config, input_channels, output_channels, 
                                            kernel_size=1, use_stride=False, use_bias=False, 
                                            use_activation=False, use_norm=False)
            self.downsample_conv2 = ConvBlock(config, output_channels, output_channels, 
                                            kernel_size=kernel_size, use_stride=downsample, 
                                            use_bias=False, use_activation=False, use_norm=False)
        elif input_channels != output_channels:
            self.channel_match_conv = ConvBlock(config, input_channels, output_channels, 
                                              kernel_size=kernel_size, use_stride=False, 
                                              use_bias=False, use_activation=False, use_norm=False)
        else:
            self.channel_match_conv = None
        
        # Main convolution layers
        self.conv1 = ConvBlock(config, input_channels, output_channels, 
                             kernel_size=kernel_size, use_stride=downsample, 
                             use_activation=False, use_norm=False)
        self.conv2 = ConvBlock(config, output_channels, output_channels, 
                             kernel_size=kernel_size, use_stride=False, 
                             use_activation=False, use_norm=False)
    
    def forward(self, x, condition=None):
        # Store input for residual connection
        residual = x.clone()
        
        # First conv block
        if self.use_film and condition is not None:
            x = self.film_layer_1(x, condition)
        else:
            x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        # Second conv block
        if self.use_film and condition is not None:
            x = self.film_layer_2(x, condition)
        else:
            x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        # Handle residual connection
        if self.downsample:
            residual = self.downsample_conv1(residual)
            residual = self.downsample_conv2(residual)
        elif self.channel_match_conv:
            residual = self.channel_match_conv(residual)
        
        return x + residual


class ConvolutionalEncoder(nn.Module):
    """
    Stack of convolutional blocks for encoding.
    """
    def __init__(self, config, num_conv_layers):
        super().__init__()
        self.config = config
        self.use_film = getattr(config, 'film_layer', False)
        
        # Build encoder blocks
        conv_blocks = []
        
        # First layer
        if self.use_film:
            conv_blocks.append(ConvBlock(config, 1, 4))
        else:
            conv_blocks.append(ConvBlock(config, 2, 4))  # 2 channels for concatenated input
        
        # Intermediate layers with exponentially increasing channels
        for i in range(1, num_conv_layers):
            in_channels = min(64, 2**(i+1))
            out_channels = min(64, 2**(i+2))
            conv_blocks.append(ConvBlock(config, in_channels, out_channels))
        
        # Optional residual blocks
        if config.resnet:
            final_channels = min(64, 2**(num_conv_layers+1))
            for _ in range(2):
                conv_blocks.append(
                    ResidualConvBlock(config, final_channels, final_channels, 
                                    kernel_size=4, downsample=False)
                )
        
        self.conv_blocks = nn.Sequential(*conv_blocks)
    
    def forward(self, x, condition=None):
        for layer in self.conv_blocks:
            x = layer(x, condition)
        return x


class DeconvolutionalBlock(nn.Module):
    """
    Transposed convolution block for decoding.
    """
    def __init__(self, config, input_channels, output_channels, 
                 use_stride=True, is_last_layer=False):
        super().__init__()
        self.config = config
        
        # Dropout configuration
        self.use_dropout = getattr(config, 'dropout', False)
        self.dropout_rate = getattr(config, 'drop_rate', 0.2)
        
        activation_layers = []
        
        # Configure transposed convolution
        if use_stride:
            self.conv = nn.ConvTranspose1d(input_channels, output_channels, 
                                         kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv1d(input_channels, output_channels, 
                                kernel_size=4, stride=1, padding="same")
        
        # Configure activation for last layer
        if is_last_layer:
            if config.last_act == "identity":
                activation_layers.append(nn.Identity())
            elif config.last_act == "sigmoid":
                activation_layers.append(nn.Sigmoid())
        else:
            # Standard intermediate layer configuration
            if config.type == "batchnorm":
                activation_layers.append(nn.BatchNorm1d(output_channels))
            elif config.type == "instancenorm":
                activation_layers.append(nn.InstanceNorm1d(output_channels, affine=True))
            
            activation_layers.append(nn.LeakyReLU(negative_slope=0.2))
            
            if self.use_dropout:
                activation_layers.append(nn.Dropout(p=self.dropout_rate))
        
        nn.init.xavier_uniform_(self.conv.weight)
        self.activation = nn.Sequential(*activation_layers)
    
    def forward(self, x, condition=None):
        x = self.conv(x)
        return self.activation(x)


class ConvolutionalDecoder(nn.Module):
    """
    Stack of deconvolutional blocks for decoding.
    """
    def __init__(self, config, num_conv_layers):
        super().__init__()
        self.config = config
        
        deconv_blocks = []
        
        # Optional residual blocks at the beginning
        if config.resnet:
            final_channels = min(64, 2**(num_conv_layers+1))
            for _ in range(2):
                deconv_blocks.append(
                    ResidualConvBlock(config, final_channels, final_channels, 
                                    kernel_size=4, downsample=False)
                )
        
        # Intermediate deconvolution layers
        for i in range(num_conv_layers-1, 0, -1):
            in_channels = min(64, 2**(2+i))
            out_channels = min(64, 2**(2+i-1))
            deconv_blocks.append(DeconvolutionalBlock(config, in_channels, out_channels))
        
        # Penultimate layer
        deconv_blocks.append(DeconvolutionalBlock(config, 4, 4))
        
        self.deconv_blocks = nn.Sequential(*deconv_blocks)
        
        # Final layer to single channel output
        self.final_layer = DeconvolutionalBlock(config, 4, 1, use_stride=False, is_last_layer=True)
    
    def forward(self, x, condition=None):
        for layer in self.deconv_blocks:
            x = layer(x, condition)
        return self.final_layer(x)


class SharedEncoder(nn.Module):
    """
    Shared encoder that maps input to latent space parameters (mu, log_sigma^2).
    """
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = config.latent_dim
        self.num_conv_layers = config.nb_conv_layer
        
        # Convolutional feature extractor
        self.conv_net = ConvolutionalEncoder(config, self.num_conv_layers)
        
        # Linear layer to latent parameters
        feature_dim = min(64, 2**(1+self.num_conv_layers)) * (input_dim // (2**self.num_conv_layers))
        self.linear = nn.Linear(feature_dim, 2 * self.latent_dim)  # mu and log_sigma^2
    
    def forward(self, x, condition=None):
        # Extract features using convolutions
        x = self.conv_net(x, condition)
        
        # Flatten and project to latent parameters
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        
        # Split into mean and log variance
        mu = x[:, :self.latent_dim]
        log_sigma_2 = x[:, self.latent_dim:]
        
        return mu, log_sigma_2


class SharedDecoder(nn.Module):
    """
    Shared decoder that maps from latent space back to input space.
    """
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_conv_layers = config.nb_conv_layer
        self.input_dim = input_dim
        self.use_film = getattr(config, 'film_layer', False)
        
        # Deconvolutional generator
        self.deconv_net = ConvolutionalDecoder(config, self.num_conv_layers)
        
        # Linear projection from latent space
        feature_dim = min(64, 2**(1+self.num_conv_layers)) * (input_dim // (2**self.num_conv_layers))
        
        if self.use_film:
            self.linear = DenseBlock(config, self.latent_dim, feature_dim)
        else:
            # Include conditioning information
            self.linear = DenseBlock(config, 2*self.latent_dim, feature_dim)
        
        # Shape for reshaping flattened features back to conv format
        self.reshape_dims = (-1, min(64, 2**(1+self.num_conv_layers)), 
                           input_dim // (2**self.num_conv_layers))
    
    def forward(self, x, condition=None):
        # Project to feature space
        x = self.linear(x)
        
        # Reshape for deconvolution
        x = x.view(self.reshape_dims)
        
        # Generate output using deconvolutional layers
        return self.deconv_net(x, condition)


class Regressor(nn.Module):
    """
    Regression head for predicting target values from latent representations.
    """
    def __init__(self, config, num_dense_layers, use_physical_data, 
                 input_class_dim=3, num_classes=1):
        super().__init__()
        
        self.use_physical_data = use_physical_data
        
        # Determine input dimension
        if use_physical_data:
            input_dim = config.latent_dim + input_class_dim
        else:
            input_dim = config.latent_dim
        
        # Determine output dimension based on dataset
        if config.dataset == 1:
            num_outputs = num_classes
        
        # Build regression layers
        layers = []
        if num_dense_layers == 1:
            layers = [DenseBlock(config, input_dim, num_outputs, is_last_layer=True)]
        elif num_dense_layers == 2:
            layers = [DenseBlock(config, input_dim, 20, 
                               activation_function=config.activation_function)]
            
            if config.dropout_cls:
                layers.append(nn.Dropout(0.2))
            
            layers.append(DenseBlock(config, 20, num_outputs, is_last_layer=True))
        
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, x, condition=None):
        if self.use_physical_data:
            regression_input = torch.cat((x, condition), dim=1)
        else:
            regression_input = x.clone()
        
        return self.regressor(regression_input)


class VAE(nn.Module):
    """
    Conditional Variational Autoencoder with optional domain adaptation and regression.
    """
    def __init__(self, config, device, input_dim, input_class_dim=3, num_classes=1):
        super().__init__()
        
        self.config = config
        self.device = device
        self.input_dim = input_dim
        self.input_class_dim = input_class_dim
        
        # Core VAE components
        self.encoder = SharedEncoder(config, input_dim)
        self.decoder = SharedDecoder(config, input_dim)
        
        # Conditioning embeddings
        self.class_embedding_encoder = DenseBlock(config, input_class_dim, input_dim, use_norm=False)
        self.class_embedding_decoder = DenseBlock(config, input_class_dim, config.latent_dim, use_norm=False)
        
        # Configuration flags
        self.use_film = getattr(config, 'film_layer', False)
        self.use_regressor = getattr(config, 'train_with_regressor', False)
        self.model_version = getattr(config, 'model_version', 1)
        self.alpha = getattr(config, 'alpha', 0.3)
        self.regress_with_mu_only = config.train_with_regressor_with_only_mu
        self.regression_with_mu = config.regression_with_mu
        
        # Optional regressor
        if self.use_regressor:
            if config.dataset == 1:
                num_outputs = num_classes
            else:
                num_categories = 4
                num_outputs = num_classes + num_categories - 1
            
            regressor_input_dim = (config.latent_dim if self.regress_with_mu_only 
                                 else input_class_dim + config.latent_dim)
            
            layers = []
            if config.nb_dense_layer == 1:
                layers = [DenseBlock(config, regressor_input_dim, num_outputs, is_last_layer=True)]
            elif config.nb_dense_layer == 2:
                layers = [DenseBlock(config, regressor_input_dim, 20), 
                         nn.Dropout(0.2),
                         DenseBlock(config, 20, num_outputs, is_last_layer=True)]
            
            self.regressor = nn.Sequential(*layers)
        
        # Domain adversarial training (DANN)
        if self.model_version == 2:
            if config.dataset == 1:
                num_outputs = input_class_dim
            else:
                num_categories = 4
                num_outputs = input_class_dim + num_categories - 1
            
            self.num_domain_outputs = num_outputs
            
            domain_layers = []
            if config.nb_dense_layer == 1:
                domain_layers = [DenseBlock(config, config.latent_dim, num_outputs, is_last_layer=True)]
            elif config.nb_dense_layer == 2:
                domain_layers = [DenseBlock(config, config.latent_dim, 20),
                               nn.Dropout(0.2),
                               DenseBlock(config, 20, num_outputs, is_last_layer=True)]
            
            self.domain_classifier = nn.Sequential(*domain_layers)
    
    def forward(self, x, condition, type=None):
        """
        Forward pass through the conditional VAE.
        
        Args:
            x: Input data [batch_size, sequence_length]
            condition: Conditioning information [batch_size, condition_dim]
            type: Types of data for loss computation
        
        Returns:
            Reconstruction and latent parameters, with optional auxiliary outputs
        """
        # Encode to latent space
        if self.model_version == 2:
            # Domain adversarial version
            mu, log_sigma_2 = self.encoder(x.unsqueeze(1))
            z = self.reparameterize(mu, log_sigma_2)
            domain_loss = self._compute_domain_loss(z, condition, type)
        else:
            # Standard conditional encoding
            if self.use_film:
                mu, log_sigma_2 = self.encoder(x.unsqueeze(1), condition)
            else:
                # Concatenate condition embedding with input
                condition_emb = self.class_embedding_encoder(condition)
                x_conditioned = torch.cat((x.unsqueeze(1), condition_emb.unsqueeze(1)), dim=1)
                mu, log_sigma_2 = self.encoder(x_conditioned)
            
            z = self.reparameterize(mu, log_sigma_2)
        
        # Optional regression
        if self.use_regressor:
            if self.regression_with_mu:
                reg_input = mu.clone() if self.regress_with_mu_only else torch.cat((mu, condition), dim=1)
            else:
                reg_input = torch.cat((z, condition), dim=1)
            
            regression_pred = self.regressor(reg_input)
        
        # Decode from latent space
        if self.use_film:
            reconstruction = self.decoder(z, condition)[:, 0]
        else:
            # Concatenate condition embedding with latent code
            condition_emb = self.class_embedding_decoder(condition)
            z_conditioned = torch.cat((z, condition_emb), dim=1)
            reconstruction = self.decoder(z_conditioned)[:, 0]
        
        # Return appropriate outputs based on configuration
        if self.model_version == 2:
            return (reconstruction, mu, log_sigma_2), domain_loss
        elif self.use_regressor:
            return (reconstruction, mu, log_sigma_2), regression_pred
        else:
            return reconstruction, mu, log_sigma_2
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE sampling.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        
        Returns:
            Sampled latent code
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _compute_domain_loss(self, z, condition, type):
        """
        Compute domain adversarial loss for domain adaptation.
        
        Args:
            z: Latent representation
            condition: Domain labels
            type: Types of data for loss computation
        
        Returns:
            Domain classification loss
        """
        # Apply gradient reversal layer
        reversed_features = ReverseLayerF.apply(z, self.alpha)
        
        # Classify domain
        domain_predictions = self.domain_classifier(reversed_features)
        
        # Compute domain loss
        return compute_loss(domain_predictions, condition, type, self.config.dataset)
    
    def exact_reconstruction(self, x, condition):
        """
        Perform reconstruction using mean of latent distribution (no sampling).
        
        Args:
            x: Input data
            condition: Conditioning information
        
        Returns:
            Exact reconstruction and latent parameters
        """
        # Encode using mean only
        if self.model_version == 2:
            mu, log_sigma_2 = self.encoder(x.unsqueeze(1))
        else:
            if self.use_film:
                mu, log_sigma_2 = self.encoder(x.unsqueeze(1), condition)
            else:
                condition_emb = self.class_embedding_encoder(condition)
                x_conditioned = torch.cat((x.unsqueeze(1), condition_emb.unsqueeze(1)), dim=1)
                mu, log_sigma_2 = self.encoder(x_conditioned)
        
        # Optional regression with mean
        if self.use_regressor:
            reg_input = mu.clone() if self.regress_with_mu_only else torch.cat((mu, condition), dim=1)
            regression_pred = self.regressor(reg_input)
        
        # Decode using mean
        if self.use_film:
            reconstruction = self.decoder(mu, condition)[:, 0]
        else:
            condition_emb = self.class_embedding_decoder(condition)
            mu_conditioned = torch.cat((mu, condition_emb), dim=1)
            reconstruction = self.decoder(mu_conditioned)[:, 0]
        
        if self.use_regressor:
            return reconstruction, mu, log_sigma_2, regression_pred
        else:
            return reconstruction, mu, log_sigma_2


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for VAE training.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, mu, log_sigma_2):
        """
        Compute KL divergence between learned distribution and unit Gaussian.
        
        Args:
            mu: Mean of learned distribution
            log_sigma_2: Log variance of learned distribution
        
        Returns:
            KL divergence loss
        """
        kl_loss = 1 + log_sigma_2 - torch.exp(log_sigma_2) - torch.square(mu)
        kl_loss = -0.5 * torch.sum(kl_loss, dim=1)
        return torch.mean(kl_loss)


class VAELoss(nn.Module):
    """
    Standard VAE loss combining reconstruction and KL divergence terms.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.reconstruction_loss = nn.MSELoss(reduction="none")
        self.kl_loss = KLDivergenceLoss()
    
    def forward(self, epoch, vae_output, target):
        """
        Compute total VAE loss.
        
        Args:
            epoch: Current training epoch (unused in this version)
            vae_output: Tuple of (reconstruction, mu, log_sigma_2)
            target: Target data
        
        Returns:
            Total loss, reconstruction loss, KL loss
        """
        reconstruction, mu, log_sigma_2 = vae_output
        
        batch_size = reconstruction.size(0)
        recon_loss = self.reconstruction_loss(reconstruction, target) / batch_size
        kl_loss = self.kl_loss(mu, log_sigma_2)
        
        total_loss = self.beta * recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def create_cyclic_schedule(num_iterations, start=0.0, stop=1.0, num_cycles=4, ratio=0.2):
    """
    Create a cyclic annealing schedule for beta parameter.
    
    Args:
        num_iterations: Total number of training iterations
        start: Starting value for schedule
        stop: Maximum value for schedule
        num_cycles: Number of annealing cycles
        ratio: Fraction of cycle spent increasing
    
    Returns:
        Array of beta values for each iteration
    """
    schedule = np.ones(num_iterations) * stop
    period = num_iterations / num_cycles
    step = (stop - start) / (period * ratio)
    
    for cycle in range(num_cycles):
        value, iteration = start, 0
        while value <= stop and (int(iteration + cycle * period) < num_iterations):
            schedule[int(iteration + cycle * period)] = value
            value += step
            iteration += 1
    
    return schedule


class VAELossWithSchedule(nn.Module):
    """
    VAE loss with cyclic annealing schedule for beta parameter.
    """
    def __init__(self, num_iterations, num_cycles=1, ratio=0.3, start=0.0, stop=1.0, 
                 loss="mse"):
        super().__init__()
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        
        if loss == "mse":
            self.reconstruction_loss = nn.MSELoss(reduction="none")
        elif loss == "mae":
            self.reconstruction_loss = nn.L1Loss(reduction="none")
        else:
            self.reconstruction_loss = nn.BCELoss(reduction="none")
        
        self.kl_loss = KLDivergenceLoss()
        
        # Create beta annealing schedule
        self.beta_schedule = create_cyclic_schedule(
            num_iterations, num_cycles=num_cycles, ratio=ratio, start=start, stop=stop
        )
            
    def forward(self, epoch, vae_output, target):
        """
        Compute VAE loss with scheduled beta parameter.
        
        Args:
            epoch: Current training epoch
            vae_output: Tuple of (reconstruction, mu, log_sigma_2)
            target: Target data
        
        Returns:
            Total loss, reconstruction loss, KL loss
        """
        reconstruction, mu, log_sigma_2 = vae_output
        
        batch_size = reconstruction.size(0)
        beta = self.beta_schedule[epoch]
        
        # Weighted reconstruction loss (emphasizes higher magnitude targets)
        reconstruction_loss = (torch.sum(torch.exp(target + 1) * 
                                       self.reconstruction_loss(reconstruction, target)) / 
                             batch_size)
        
        kl_loss = self.kl_loss(mu, log_sigma_2)
        total_loss = reconstruction_loss + beta * kl_loss
        
        return total_loss, reconstruction_loss, kl_loss
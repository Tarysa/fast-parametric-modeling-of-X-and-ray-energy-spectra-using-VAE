# fast-parametric-modeling-of-X-and-ray-energy-spectra-using-VAE


## I. Meleshenkovskii, T. Sendra, E. Mauerhofer, V. Vigneron

Official implementation of **A new generative framework for fast parametric modeling of X− and γ−ray energy spectra using VAE**

📁 Project Structure

```text

code/
├── datasets/              # Input datasets
│   └── ESARDA/
│   └── MC/
├── models/            # Model architectures and related modules
│   └── GRL.py  
│   └── model.py   
├── outputs/      # Generated outputs (e.g., predictions, logs)
├── runs/      # Checkpoints or run metadata
├── train_eval/      # Training and evaluation scripts
│   └── regressor.py  
│   └── script.py   
│   └── test.py  
│   └── train.py  
├── utils/               # Utility functions and preprocessing
│   └── functions.py  
│   └── preprocessing.py  
├── weights/          # Pretrained or trained model weights

```

🚀 Getting Started

### 1. Install dependencies

Recommended: Create and activate a virtual environment

You can install dependencies with:

pip install -r requirements.txt

### 2. Training

To train a model, use one of the following script:

```text
python3 train.py --dataset 1 --beta 2 --lr 1e-4 --n_epochs  20000 --latent_dim 10 --opt "adamw" --batch_size 4  --resnet True --nb_conv_layer 3 --reduction "sum" --resnet True --loss "mse" --version 3 --beta_schedule_ratio 0.4 --weight_decay 5e-2 --film_layer True --beta_regressor 1 --type "batchnorm"  --nb_dense_layer 2 --model_version 2 --alpha 0.6
```

```text
python3 train.py --dataset 3 --beta 0.1 --lr 1e-4 --n_epochs  20000 --latent_dim 10 --opt "adamw" --batch_size 6  --resnet True --nb_conv_layer 3 --reduction "sum" --resnet True --loss "mse" --version 3 --beta_schedule_ratio 0.4 --weight_decay 5e-2 --film_layer True --beta_regressor 0.5 --type "batchnorm"  --nb_dense_layer 2 --reduced_dataset True --model_version 2 --alpha 1
```

### 3. Testing

Evaluate the model with:

```text
python3 test.py --dataset 1
```

```text
python3 test.py --dataset 3
```

Make sure the corresponding model weights are available in the weights/ directory.

### License

### Author

```text
I. Meleshenkovskii, T. Sendra, E. Mauerhofer, V. Vigneron
```
=======

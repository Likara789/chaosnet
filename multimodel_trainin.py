"""
Multi-Modal Training Script for ChaosNet
---------------------------------------
This script implements a multi-modal learning system using ChaosNet architecture.
It supports training on different types of data (text, images) with shared or separate
neural components. The implementation includes data loading, model definition,
training loops, and evaluation metrics.

Key Components:
- Text classification using AG News dataset
- Image classification using MNIST and EMNIST datasets
- Shared ChaosCortex for cross-modal learning
- Training and evaluation utilities

Author: Likara789
Email: lowkeytripping.dev@gmail.com
"""

import csv
import json
import pickle
import re
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex

# Constants for text processing and model configuration
PAD_TOKEN = "<pad>"  # Padding token for sequence alignment
UNK_TOKEN = "<unk>"  # Token for unknown/out-of-vocabulary words
PAD_IDX = 0          # Index for padding token in vocabulary
UNK_IDX = 1          # Index for unknown token in vocabulary

# Dataset-specific constants
NUM_AG_CLASSES = 4        # Number of classes in AG News dataset
NUM_MNIST_CLASSES = 10    # Number of classes in MNIST (digits 0-9)
NUM_EMNIST_CLASSES = 26   # Number of classes in EMNIST (letters A-Z)

# Dataset URLs and paths
AG_NEWS_URL = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
AG_NEWS_DIR = Path("./data/ag_news")
TOKEN_REGEX = re.compile(r"\w+")


def rotate_emnist(image):
    return torch.rot90(image, 1, dims=(1, 2))


class LanguageDataset(Dataset):
    """
    A PyTorch Dataset class for handling text data for language modeling.
    
    This class handles tokenization, numericalization, and padding of text data
    to create fixed-length sequences suitable for neural network processing.
    
    Args:
        samples (list): List of (text, label) tuples
        vocab (dict): Vocabulary mapping tokens to indices
        tokenizer (callable): Function to split text into tokens
        max_len (int): Maximum sequence length (longer sequences will be truncated)
    """
    def __init__(self, samples, vocab, tokenizer, max_len=128):
        """
        Initialize the LanguageDataset instance.
        
        Args:
            samples (list): List of (text, label) tuples
            vocab (dict): Vocabulary mapping tokens to indices
            tokenizer (callable): Function to split text into tokens
            max_len (int): Maximum sequence length (longer sequences will be truncated)
        """
        self.samples = samples
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (tokens_tensor, label_tensor) where:
                - tokens_tensor: Padded sequence of token indices
                - label_tensor: Class label as a tensor
        """
        label, text = self.samples[idx]
        tokens = self.tokenizer(text)[: self.max_len]
        indices = [self.vocab.get(token, UNK_IDX) for token in tokens]
        if len(indices) < self.max_len:
            indices += [PAD_IDX] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), label - 1


def build_vocab(samples, tokenizer, min_freq=5, max_tokens=20000):
    """
    Build a vocabulary from the given text samples.
    
    Args:
        samples (list): List of (label, text) tuples
        tokenizer (callable): Function to split text into tokens
        min_freq (int): Minimum frequency for a token to be included
        max_tokens (int): Maximum vocabulary size (most frequent tokens)
        
    Returns:
        dict: Vocabulary mapping tokens to indices
    """
    counter = Counter()
    # Count token frequencies across all samples
    for _, text in samples:
        counter.update(tokenizer(text))
        
    # Initialize with special tokens
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    
    # Add most frequent tokens that meet minimum frequency
    for token, count in counter.most_common(max_tokens):
        if count >= min_freq:
            vocab[token] = len(vocab)
            
    return vocab


def download_ag_news():
    """
    Download and extract the AG News dataset if it doesn't exist locally.
    
    The dataset is downloaded from a predefined URL and extracted to the
    directory specified by AG_NEWS_DIR.
    """
    if not AG_NEWS_DIR.exists():
        AG_NEWS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading AG News dataset to {AG_NEWS_DIR}...")
        filename, _ = urllib.request.urlretrieve(AG_NEWS_URL)
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(AG_NEWS_DIR.parent)
        print("Download complete.")
    else:
        print(f"AG News dataset already exists at {AG_NEWS_DIR}")


def read_ag_news_split(data_root, split):
    """
    Read a split of the AG News dataset.
    
    Args:
        data_root (Path): Root directory containing the dataset
        split (str): Dataset split to read ('train' or 'test')
        
    Returns:
        list: List of (label, text) tuples
    """
    filename = data_root / f"{split}.csv"
    samples = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            label, title, description = int(row[0]), row[1], row[2]
            text = title + ' ' + description  # Combine title and description
            samples.append((label, text))
    return samples


def simple_tokenizer(text):
    """
    Simple tokenizer that splits text into lowercase words.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of lowercase word tokens
    """
    return TOKEN_REGEX.findall(text.lower())


def prepare_ag_news_dataloaders(batch_size=64, max_seq_len=128, val_split=0.1):
    """
    Prepare DataLoaders for AG News dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        max_seq_len (int): Maximum sequence length (longer sequences will be truncated)
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab) where:
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - test_loader: DataLoader for test data
            - vocab: Vocabulary built from training data
    """
    # Download and load dataset splits
    data_root = download_ag_news()
    train_samples = read_ag_news_split(data_root, "train")
    test_samples = read_ag_news_split(data_root, "test")

    # Build vocabulary from training data
    vocab = build_vocab(train_samples, simple_tokenizer)
    
    # Split training data into train/validation sets
    val_size = int(len(train_samples) * val_split)
    train_samples, val_samples = random_split(
        train_samples, [len(train_samples) - val_size, val_size]
    )

    # Create dataset objects
    train_dataset = LanguageDataset(train_samples, vocab, simple_tokenizer, max_seq_len)
    val_dataset = LanguageDataset(val_samples, vocab, simple_tokenizer, max_seq_len)
    test_dataset = LanguageDataset(test_samples, vocab, simple_tokenizer, max_seq_len)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader, vocab


def prepare_mnist_dataloaders(batch_size=128, val_split=0.1):
    """
    Prepare DataLoaders for MNIST dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoaders for MNIST
    """
    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    # Split training data into train/validation sets
    val_size = int(len(train_dataset) * val_split)
    train_dataset, val_dataset = random_split(
        train_dataset, [len(train_dataset) - val_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


def prepare_emnist_dataloaders(batch_size=128, val_split=0.1):
    """
    Prepare DataLoaders for EMNIST Letters dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoaders for EMNIST Letters
    """
    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize((0.1736,), (0.3317,)),  # EMNIST Letters mean and std
            transforms.Lambda(rotate_emnist),  # Rotate images to correct orientation
        ]
    )

    # Load EMNIST Letters dataset
    train_dataset = datasets.EMNIST(
        "./data", split="letters", train=True, download=True, transform=transform
    )
    test_dataset = datasets.EMNIST(
        "./data", split="letters", train=False, download=True, transform=transform
    )

    # Filter out empty classes (some EMNIST letters have no test samples)
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label > 0]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label > 0]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Adjust labels to be 0-25 (A-Z)
    for dataset in [train_dataset, test_dataset]:
        for i in range(len(dataset)):
            if hasattr(dataset, 'dataset'):
                dataset.dataset.targets[dataset.indices[i]] -= 1
            else:
                dataset.targets[i] -= 1

    # Split training data into train/validation sets
    val_size = int(len(train_dataset) * val_split)
    train_dataset, val_dataset = random_split(
        train_dataset, [len(train_dataset) - val_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


class ChaosLanguageModel(nn.Module):
    """
    A language model using ChaosNet architecture for text classification.
    
    This model processes sequences of tokens through an embedding layer, a ChaosCortex
    for temporal processing, and a linear readout layer for classification.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embed_dim (int): Dimensionality of token embeddings
        hidden (int): Number of hidden units in the ChaosCortex
        ticks (int): Number of time steps to run the ChaosCortex
        shared_cortex (ChaosCortex, optional): Pre-initialized ChaosCortex to share between models
        **cortex_kwargs: Additional arguments for ChaosNeuronParams
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden=256,
        ticks=5,
        shared_cortex=None,
        **cortex_kwargs
    ):
        super().__init__()
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # Use shared cortex if provided, otherwise create a new one
        if shared_cortex is not None:
            self.cortex = shared_cortex
        else:
            # Configure neuron parameters for the chaos network
            neuron_params = ChaosNeuronParams(
                threshold=0.5,               # Spiking threshold
                noise_std=0.01,              # Standard deviation of noise
                fail_prob=cortex_kwargs.get('fail_prob', 0.5),  # Probability of spike failure
                decay=0.02,                  # Membrane potential decay rate
                refractory_decay=0.95,       # Refractory period decay rate
            )
            # Configure cortex (network) parameters
            cortex_params = CortexParams(
                input_size=embed_dim,        # Input dimension matches embedding size
                hidden_sizes=[hidden],       # Single hidden layer
                neuron=neuron_params,        # Neuron configuration
            )
            self.cortex = ChaosCortex(cortex_params)

        # Output layer for classification
        self.readout = nn.Linear(hidden, NUM_AG_CLASSES)
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.readout.weight, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.readout.bias, 0.0)

        self.ticks = ticks  # Number of time steps to run the network

    def forward(self, tokens, collect_spikes=False):
        emb = self.embedding(tokens)
        mask = (tokens != PAD_IDX).unsqueeze(-1)
        valid = mask.sum(dim=1).clamp(min=1)
        avg_emb = (emb * mask).sum(dim=1) / valid

        batch = tokens.size(0)
        ticks = self.ticks
        expanded = avg_emb.unsqueeze(1).expand(batch, ticks, -1).reshape(batch * ticks, -1)

        out, _, layer_spikes = self.cortex(expanded, None)
        spikes = layer_spikes[0] if layer_spikes else out
        spikes = spikes.view(batch, ticks, -1)

        avg_spikes = spikes.mean(dim=1)
        logits = self.readout(avg_spikes)

        if collect_spikes:
            spike_stack = spikes.permute(1, 0, 2).detach()
            return logits, spike_stack
        return logits, avg_spikes.unsqueeze(0)


class ChaosVisionModel(nn.Module):
    """
    A vision model using ChaosNet architecture for image classification.
    
    This model processes images through a CNN feature extractor, projects the features
    to a lower dimension, processes them through a ChaosCortex, and produces class scores.
    
    Args:
        embed_dim (int): Dimensionality of the projected features
        hidden (int): Number of hidden units in the ChaosCortex
        ticks (int): Number of time steps to run the ChaosCortex
        num_classes (int): Number of output classes
        shared_cortex (ChaosCortex, optional): Pre-initialized ChaosCortex to share between models
        **cortex_kwargs: Additional arguments for ChaosNeuronParams
    """
    def __init__(
        self,
        embed_dim=128,
        hidden=256,
        ticks=5,
        num_classes=NUM_MNIST_CLASSES,
        shared_cortex=None,
        **cortex_kwargs,
    ):
        super().__init__()
        # CNN-based feature extractor
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Preserve spatial dimensions
            nn.BatchNorm2d(32),  # Normalize activations
            nn.ReLU(inplace=True),  # Non-linearity
            nn.MaxPool2d(2),  # Downsample by factor of 2
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Final feature map size: 7x7 for 28x28 input
        )
        
        # Project features to lower dimension for chaos processing
        self.projection = nn.Sequential(
            nn.Flatten(),  # Flatten spatial dimensions
            nn.Linear(64 * 7 * 7, embed_dim),  # Project to embed_dim
            nn.ReLU(inplace=True),  # Non-linearity
        )

        # Use shared cortex if provided, otherwise create a new one
        if shared_cortex is not None:
            self.cortex = shared_cortex
        else:
            # Configure neuron parameters for the chaos network
            neuron_params = ChaosNeuronParams(
                threshold=0.5,               # Spiking threshold
                noise_std=0.01,              # Standard deviation of noise
                fail_prob=cortex_kwargs.get('fail_prob', 0.5),  # Probability of spike failure
                decay=0.02,                  # Membrane potential decay rate
                refractory_decay=0.95,       # Refractory period decay rate
            )
            # Configure cortex (network) parameters
            cortex_params = CortexParams(
                input_size=embed_dim,        # Input dimension matches projection size
                hidden_sizes=[hidden],       # Single hidden layer
                neuron=neuron_params,        # Neuron configuration
            )
            self.cortex = ChaosCortex(cortex_params)

        # Output layer for classification
        self.readout = nn.Linear(hidden, num_classes)
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.readout.weight, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.readout.bias, 0.0)

        self.ticks = ticks  # Number of time steps to run the network

    def forward(self, images, collect_spikes=False):
        """
        Forward pass of the vision model.
        
        Args:
            images (torch.Tensor): Input images [batch_size, 1, height, width]
            collect_spikes (bool): Whether to collect and return spike information
            
        Returns:
            torch.Tensor: Output logits [batch_size, num_classes]
            torch.Tensor: (Optional) Spike information if collect_spikes=True
        """
        emb = self.projection(self.feature_extractor(images))

        batch = images.size(0)
        ticks = self.ticks
        expanded = emb.unsqueeze(1).expand(batch, ticks, -1).reshape(batch * ticks, -1)

        out, _, layer_spikes = self.cortex(expanded, None)
        spikes = layer_spikes[0] if layer_spikes else out
        spikes = spikes.view(batch, ticks, -1)

        avg_spikes = spikes.mean(dim=1)
        logits = self.readout(avg_spikes)

        if collect_spikes:
            spike_stack = spikes.permute(1, 0, 2).detach()
            return logits, spike_stack
        return logits, avg_spikes.unsqueeze(0)


def train_epoch(model, dataloader, optimizer, device, scheduler=None, accumulation_steps=1):
    model.train()
    total = correct = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _ = model(inputs)
        loss = F.cross_entropy(outputs, targets) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item() * accumulation_steps

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    total = correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(dataloader), correct / total


def run_retention_evaluation(task_name, model, dataloader, device, runs=3):
    metrics = []
    for run in range(1, runs + 1):
        loss, acc = evaluate(model, dataloader, device)
        metrics.append({"run": run, "loss": loss, "acc": acc})
        print(
            f"[Retention] {task_name.upper()} | Run {run}/{runs} | "
            f"Loss: {loss:.4f} | Acc: {acc*100:.2f}%"
        )

    avg_loss = sum(m["loss"] for m in metrics) / runs
    avg_acc = sum(m["acc"] for m in metrics) / runs
    return {"task": task_name, "avg_loss": avg_loss, "avg_acc": avg_acc, "runs": metrics}


def save_retention_report(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def summarize_retention(baseline, final):
    summary = []
    for task in baseline:
        base = baseline[task]["avg_acc"]
        fin = final[task]["avg_acc"]
        summary.append(
            f"{task.upper()} retention | Acc {base*100:.2f}% -> {fin*100:.2f}% | "
            f"Delta {((fin - base) * 100):+.2f} pts"
        )
    return summary


def run_retention_suite(mapping, device, runs=3):
    return {
        label: run_retention_evaluation(label, model, dataloader, device, runs=runs)
        for label, (model, dataloader) in mapping.items()
    }

def setup_experiment(base="experiments/multimodal"):
    exp_dir = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_training_log(train_losses, train_accs, val_losses, val_accs, path):
    log_data = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
    }
    with open(path, "wb") as f:
        pickle.dump(log_data, f)


def train_task(
    task_name: str,
    model: nn.Module,
    dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    exp_dir: Path,
    epochs: int,
    accumulation_steps: int = 1,
    checkpoint_interval: int = 5,
):
    task_dir = exp_dir / task_name
    ckpt_dir = task_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl = dataloaders
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_dl,
            optimizer,
            device,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
        )
        val_loss, val_acc = evaluate(model, val_dl, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / f"best_{task_name}_model.pt")

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            test_loss, test_acc = evaluate(model, test_dl, device)
            print(
                f"{task_name.upper()} | Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, ckpt_dir / f"{task_name}_epoch_{epoch}.pt")
            save_training_log(
                train_losses,
                train_accs,
                val_losses,
                val_accs,
                path=task_dir / "training_log.pkl",
            )
        else:
            print(
                f"{task_name.upper()} | Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Acc: {val_acc*100:.2f}%"
            )

    torch.save(model.state_dict(), task_dir / f"final_{task_name}_model.pt")
    save_training_log(
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        path=task_dir / "training_log.pkl",
    )
    print(
        f"Finished training {task_name.upper()} | Best validation accuracy: {best_val_acc*100:.2f}% "
        f"| Artifacts saved to {task_dir}"
    )
    return best_val_acc


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = setup_experiment()
    print(f"Experiment directory: {exp_dir}")

    ag_train_dl, ag_val_dl, ag_test_dl, vocab_size = prepare_ag_news_dataloaders()
    mnist_train_dl, mnist_val_dl, mnist_test_dl = prepare_mnist_dataloaders()
    emnist_train_dl, emnist_val_dl, emnist_test_dl = prepare_emnist_dataloaders()

    # Shared cortex parameters
    shared_hidden = 512
    shared_embed_dim = 128
    shared_ticks = 5
    shared_fail_prob = 0.3  # Using the more conservative fail_prob

    # Create shared cortex
    neuron_params = ChaosNeuronParams(
        threshold=0.5,
        noise_std=0.01,
        fail_prob=shared_fail_prob,
        decay=0.02,
        refractory_decay=0.95,
    )
    cortex_params = CortexParams(
        input_size=shared_embed_dim,
        hidden_sizes=[shared_hidden],
        neuron=neuron_params,
    )
    shared_cortex = ChaosCortex(cortex_params).to(device)

    # Initialize models with shared cortex
    ag_model = ChaosLanguageModel(
        vocab_size=vocab_size,
        embed_dim=shared_embed_dim,
        hidden=shared_hidden,
        ticks=shared_ticks,
        shared_cortex=shared_cortex,
    ).to(device)

    mnist_model = ChaosVisionModel(
        embed_dim=shared_embed_dim,
        hidden=shared_hidden,
        ticks=shared_ticks,
        shared_cortex=shared_cortex,
    ).to(device)

    emnist_model = ChaosVisionModel(
        embed_dim=shared_embed_dim,
        hidden=shared_hidden,
        ticks=shared_ticks,
        num_classes=NUM_EMNIST_CLASSES,
        shared_cortex=shared_cortex,
    ).to(device)

    retention_dir = exp_dir / "retention_checks"
    retention_dir.mkdir(parents=True, exist_ok=True)

    retention_models = {
        "ag_news": (ag_model, ag_test_dl),
        "mnist": (mnist_model, mnist_test_dl),
        "emnist_letters": (emnist_model, emnist_test_dl),
    }

    baseline_results = run_retention_suite(retention_models, device, runs=3)
    save_retention_report({"baseline": baseline_results}, retention_dir / "baseline.json")

    ag_optimizer = optim.AdamW(ag_model.parameters(), lr=5e-4, weight_decay=1e-4)
    ag_scheduler = optim.lr_scheduler.OneCycleLR(
        ag_optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(ag_train_dl),
        epochs=4,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    mnist_optimizer = optim.AdamW(mnist_model.parameters(), lr=1e-3, weight_decay=1e-4)
    mnist_scheduler = optim.lr_scheduler.OneCycleLR(
        mnist_optimizer,
        max_lr=2e-3,
        steps_per_epoch=len(mnist_train_dl),
        epochs=4,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    emnist_optimizer = optim.AdamW(emnist_model.parameters(), lr=1e-3, weight_decay=1e-4)
    emnist_scheduler = optim.lr_scheduler.OneCycleLR(
        emnist_optimizer,
        max_lr=2e-3,
        steps_per_epoch=len(emnist_train_dl),
        epochs=5,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    ag_best = train_task(
        "ag_news",
        ag_model,
        (ag_train_dl, ag_val_dl, ag_test_dl),
        ag_optimizer,
        ag_scheduler,
        device,
        exp_dir,
        epochs=4,
        accumulation_steps=4,
        checkpoint_interval=5,
    )

    mnist_best = train_task(
        "mnist",
        mnist_model,
        (mnist_train_dl, mnist_val_dl, mnist_test_dl),
        mnist_optimizer,
        mnist_scheduler,
        device,
        exp_dir,
        epochs=4,
        accumulation_steps=1,
        checkpoint_interval=3,
    )

    emnist_best = train_task(
        "emnist_letters",
        emnist_model,
        (emnist_train_dl, emnist_val_dl, emnist_test_dl),
        emnist_optimizer,
        emnist_scheduler,
        device,
        exp_dir,
        epochs=5,
        accumulation_steps=1,
        checkpoint_interval=3,
    )

    print(
        f"\nTraining complete for all tasks. "
        f"Best Val Acc - AG_NEWS: {ag_best*100:.2f}% | MNIST: {mnist_best*100:.2f}% "
        f"| EMNIST: {emnist_best*100:.2f}%. See {exp_dir} for checkpoints and logs."
    )

    final_results = run_retention_suite(retention_models, device, runs=3)
    save_retention_report({"final": final_results}, retention_dir / "final.json")
    for line in summarize_retention(baseline_results, final_results):
        print(line)


if __name__ == "__main__":
    main()







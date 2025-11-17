import csv
import json
import pickle
import re
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import datasets, transforms

from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex
from typing import Optional, Dict, Any, List

# =============================================
# Configuration Section
# =============================================

@dataclass
class DataConfig:
    # AG News dataset configuration
    ag_news_url: str = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
    ag_news_dir: str = "./data/ag_news"
    max_seq_len: int = 128
    min_freq: int = 5
    max_tokens: int = 20000
    val_split: float = 0.1
    
    # Image datasets configuration
    image_size: int = 28
    image_channels: int = 1
    
    # DataLoader configuration
    batch_sizes: Dict[str, int] = None  # Will be initialized in __post_init__
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = {
                'ag_news': 64,
                'mnist': 128,
                'emnist': 128
            }


@dataclass
class ModelConfig:
    # Shared model parameters
    shared_embed_dim: int = 16
    shared_hidden: int = 64
    shared_ticks: int = 10
    
    # Vision model specific
    vision_features: List[int] = None  # Will be initialized in __post_init__
    
    # Language model specific
    vocab_size: int = 0  # Will be set based on dataset
    
    # Neuron parameters
    threshold: float = 0.5
    noise_std: float = 0.01
    fail_prob: float = 0.5
    decay: float = 0.02
    refractory_decay: float = 0.95
    
    def __post_init__(self):
        if self.vision_features is None:
            self.vision_features = [16, 32]  # Feature channels for conv layers


def create_cortex(input_dim: int, hidden_dim: int, config: ModelConfig, shared_cortex=None):
    """Create a cortex with consistent parameters from config."""
    if shared_cortex is not None:
        return shared_cortex
        
    neuron_params = ChaosNeuronParams(
        threshold=config.threshold,
        noise_std=config.noise_std,
        fail_prob=config.fail_prob,
        decay=config.decay,
        refractory_decay=config.refractory_decay,
    )
    
    cortex_params = CortexParams(
        input_size=input_dim,
        hidden_sizes=[hidden_dim],
        neuron=neuron_params,
    )
    
    return ChaosCortex(cortex_params)



@dataclass
class TrainingConfig:
    # General training parameters
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_interval: int = 5
    accumulation_steps: Dict[str, int] = None  # Will be initialized in __post_init__
    
    # Task-specific configurations
    tasks: Dict[str, Dict[str, Any]] = None  # Will be initialized in __post_init__
    
    # Experiment tracking
    experiment_base: str = "experiments/multimodal"
    save_retention_reports: bool = True
    retention_runs: int = 3
    
    def __post_init__(self):
        if self.accumulation_steps is None:
            self.accumulation_steps = {
                'ag_news': 4,
                'mnist': 1,
                'emnist_letters': 1
            }
            
        if self.tasks is None:
            self.tasks = {
                'ag_news': {
                    'epochs': 4,
                    'optimizer': {
                        'lr': 5e-4,
                        'weight_decay': 1e-4,
                        'betas': (0.9, 0.999),
                        'eps': 1e-8
                    },
                    'scheduler': {
                        'max_lr': 1e-3,
                        'pct_start': 0.3,
                        'anneal_strategy': 'cos'
                    }
                },
                'mnist': {
                    'epochs': 4,
                    'optimizer': {
                        'lr': 1e-3,
                        'weight_decay': 1e-4,
                        'betas': (0.9, 0.999),
                        'eps': 1e-8
                    },
                    'scheduler': {
                        'max_lr': 2e-3,
                        'pct_start': 0.2,
                        'anneal_strategy': 'cos'
                    }
                },
                'emnist_letters': {
                    'epochs': 5,
                    'optimizer': {
                        'lr': 1e-3,
                        'weight_decay': 1e-4,
                        'betas': (0.9, 0.999),
                        'eps': 1e-8
                    },
                    'scheduler': {
                        'max_lr': 2e-3,
                        'pct_start': 0.2,
                        'anneal_strategy': 'cos'
                    }
                }
            }


# Initialize configurations
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()

# Constants (these typically don't need to be changed)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1
NUM_AG_CLASSES = 4
NUM_MNIST_CLASSES = 10
NUM_EMNIST_CLASSES = 26
AG_NEWS_URL = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
AG_NEWS_DIR = Path("./data/ag_news")
TOKEN_REGEX = re.compile(r"\w+")


def rotate_emnist(image):            
    return torch.rot90(image, 1, dims=(1, 2))


class LanguageDataset(Dataset):
    def __init__(self, samples, vocab, tokenizer, max_len=128):
        self.samples = samples
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        tokens = self.tokenizer(text)[: self.max_len]
        indices = [self.vocab.get(token, UNK_IDX) for token in tokens]
        if len(indices) < self.max_len:
            indices += [PAD_IDX] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), label - 1


def build_vocab(samples, tokenizer, min_freq=5, max_tokens=20000):
    counter = Counter()
    for _, text in samples:
        counter.update(tokenizer(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_tokens:
            break
        vocab[token] = len(vocab)

    return vocab


def download_ag_news():
    if not AG_NEWS_DIR.exists():
        AG_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    data_root = AG_NEWS_DIR / "ag_news_csv"
    if data_root.exists():
        return data_root

    tar_path = AG_NEWS_DIR / "ag_news_csv.tgz"
    if not tar_path.exists():
        print("Downloading AG_NEWS dataset...")
        urllib.request.urlretrieve(AG_NEWS_URL, tar_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=AG_NEWS_DIR)

    return data_root


def read_ag_news_split(data_root, split):
    path = data_root / f"{split}.csv"
    samples = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for label, title, desc in reader:
            text = f"{title} {desc}"
            samples.append((int(label), text))
    return samples


def simple_tokenizer(text):
    return TOKEN_REGEX.findall(text.lower())


def prepare_ag_news_dataloaders(batch_size=64, max_seq_len=128, val_split=0.1):
    data_root = download_ag_news()
    train_samples = read_ag_news_split(data_root, "train")
    test_samples = read_ag_news_split(data_root, "test")

    vocab = build_vocab(train_samples, simple_tokenizer)
    val_size = int(len(train_samples) * val_split)
    train_slice, val_slice = (
        train_samples[val_size:],
        train_samples[:val_size],
    )

    train_ds = LanguageDataset(train_slice, vocab, simple_tokenizer, max_seq_len)
    val_ds = LanguageDataset(val_slice, vocab, simple_tokenizer, max_seq_len)
    test_ds = LanguageDataset(test_samples, vocab, simple_tokenizer, max_seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_dl, val_dl, test_dl, len(vocab)


def prepare_mnist_dataloaders(batch_size=128, val_split=0.1):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_dl = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_subset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl


def prepare_emnist_dataloaders(batch_size=128, val_split=0.1):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rotate_emnist),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.EMNIST(
        "./data",
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.EMNIST(
        "./data",
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset.targets = train_dataset.targets - 1
    test_dataset.targets = test_dataset.targets - 1

    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(123)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_dl = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_subset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl


class ChaosLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        config: ModelConfig,
        shared_cortex: Optional[ChaosCortex] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.shared_embed_dim, padding_idx=PAD_IDX)
        
        # Create or use shared cortex
        self.cortex = create_cortex(
            input_dim=config.shared_embed_dim,
            hidden_dim=config.shared_hidden,
            config=config,
            shared_cortex=shared_cortex
        )
        
        self.readout = nn.Linear(config.shared_hidden, NUM_AG_CLASSES)
        nn.init.kaiming_normal_(self.readout.weight, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.readout.bias, 0.0)
        
        self.ticks = config.shared_ticks

    def forward(self, tokens, collect_spikes=False):
        emb = self.embedding(tokens)
        mask = (tokens != PAD_IDX).unsqueeze(-1)
        valid = mask.sum(dim=1).clamp(min=1)
        avg_emb = (emb * mask).sum(dim=1) / valid

        batch = tokens.size(0)
        state = self.cortex.init_state(batch, device=tokens.device, dtype=avg_emb.dtype)

        spikes_accum = torch.zeros(
            batch,
            self.readout.in_features,
            device=avg_emb.device,
            dtype=avg_emb.dtype,
        )
        spikes_list = [] if collect_spikes else None

        for _ in range(self.ticks):
            out, state, layer_spikes = self.cortex(avg_emb, state)
            spikes = layer_spikes[0] if layer_spikes else out
            spikes_accum.add_(spikes)
            if collect_spikes:
                spikes_list.append(spikes.detach())

        avg_spikes = spikes_accum / self.ticks
        logits = self.readout(avg_spikes)

        if collect_spikes:
            return logits, torch.stack(spikes_list)
        return logits, avg_spikes.unsqueeze(0)


class ChaosVisionModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_classes: int = NUM_MNIST_CLASSES,
        shared_cortex: Optional[ChaosCortex] = None,
    ):
        super().__init__()
        # Feature extractor using configurable feature channels
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, config.vision_features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(config.vision_features[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(config.vision_features[0], config.vision_features[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(config.vision_features[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Calculate flattened size after convolutions and pooling
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.vision_features[1] * 7 * 7, config.shared_embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Create or use shared cortex
        self.cortex = create_cortex(
            input_dim=config.shared_embed_dim,
            hidden_dim=config.shared_hidden,
            config=config,
            shared_cortex=shared_cortex
        )
        
        self.readout = nn.Linear(config.shared_hidden, num_classes)
        nn.init.kaiming_normal_(self.readout.weight, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.readout.bias, 0.0)
        
        self.ticks = config.shared_ticks

    def forward(self, images, collect_spikes=False):
        emb = self.projection(self.feature_extractor(images))

        batch = images.size(0)
        state = self.cortex.init_state(batch, device=images.device, dtype=emb.dtype)

        spikes_accum = torch.zeros(
            batch,
            self.readout.in_features,
            device=emb.device,
            dtype=emb.dtype,
        )
        spikes_list = [] if collect_spikes else None

        for _ in range(self.ticks):
            out, state, layer_spikes = self.cortex(emb, state)
            spikes = layer_spikes[0] if layer_spikes else out
            spikes_accum.add_(spikes)
            if collect_spikes:
                spikes_list.append(spikes.detach())

        avg_spikes = spikes_accum / self.ticks
        logits = self.readout(avg_spikes)

        if collect_spikes:
            return logits, torch.stack(spikes_list)
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
    shared_hidden = 128
    shared_embed_dim = 32
    shared_ticks = 10
    shared_fail_prob = 0.5  # Using the more conservative fail_prob

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

    # Create model config
    model_config = ModelConfig(
        shared_embed_dim=shared_embed_dim,
        shared_hidden=shared_hidden,
        shared_ticks=shared_ticks,
        vision_features=[32, 64]  # Default values for vision model
    )

    # Initialize models with shared cortex
    ag_model = ChaosLanguageModel(
        vocab_size=vocab_size,
        config=model_config,
        shared_cortex=shared_cortex,
    ).to(device)

    mnist_model = ChaosVisionModel(
        config=model_config,
        shared_cortex=shared_cortex,
    ).to(device)

    emnist_model = ChaosVisionModel(
        config=model_config,
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
        epochs=8,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    mnist_optimizer = optim.AdamW(mnist_model.parameters(), lr=1e-3, weight_decay=1e-4)
    mnist_scheduler = optim.lr_scheduler.OneCycleLR(
        mnist_optimizer,
        max_lr=2e-3,
        steps_per_epoch=len(mnist_train_dl),
        epochs=8,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    emnist_optimizer = optim.AdamW(emnist_model.parameters(), lr=1e-3, weight_decay=1e-4)
    emnist_scheduler = optim.lr_scheduler.OneCycleLR(
        emnist_optimizer,
        max_lr=2e-3,
        steps_per_epoch=len(emnist_train_dl),
        epochs=10,
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
        epochs=8,
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
        epochs=8,
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
        epochs=10,
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

    # Save final models at the end of the run (top-level for convenience)
    try:
        torch.save(ag_model.state_dict(), exp_dir / "final_ag_news_model.pt")
        torch.save(mnist_model.state_dict(), exp_dir / "final_mnist_model.pt")
        torch.save(emnist_model.state_dict(), exp_dir / "final_emnist_letters_model.pt")
    except Exception as e:
        print(f"Warning: failed to save final models to {exp_dir}: {e}")


if __name__ == "__main__":
    main()




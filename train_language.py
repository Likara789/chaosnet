import csv
import pickle
import re
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from chaosnet.config import ChaosNeuronParams, CortexParams
from chaosnet.core.cortex import ChaosCortex

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1
NUM_CLASSES = 4
AG_NEWS_URL = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
AG_NEWS_DIR = Path("./data/ag_news")
TOKEN_REGEX = re.compile(r"\w+")


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


def prepare_dataloaders(batch_size=64, max_seq_len=128, val_split=0.1):
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


class ChaosLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden=256,
        ticks=5,
        fail_prob=0.999,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        neuron_params = ChaosNeuronParams(
            threshold=0.5,
            noise_std=0.01,
            fail_prob=fail_prob,
            decay=0.02,
            refractory_decay=0.95,
        )
        cortex_params = CortexParams(
            input_size=embed_dim,
            hidden_sizes=[hidden],
            neuron=neuron_params,
        )

        self.cortex = ChaosCortex(cortex_params)
        self.readout = nn.Linear(hidden, NUM_CLASSES)
        nn.init.kaiming_normal_(self.readout.weight, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.readout.bias, 0.0)

        self.ticks = ticks

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


def train_epoch(model, dataloader, optimizer, device, scheduler=None, accumulation_steps=4):
    model.train()
    total = correct = running_loss = 0
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
    total = correct = running_loss = 0
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


def setup_experiment(base="experiments/language"):
    exp_dir = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
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


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = setup_experiment()
    print(f"Experiment directory: {exp_dir}")

    train_dl, val_dl, test_dl, vocab_size = prepare_dataloaders()

    model = ChaosLanguageModel(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden=512,
        ticks=5,
        fail_prob=0.5,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_dl),
        epochs=15,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    num_epochs = 15
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_dl,
            optimizer,
            device,
            scheduler=scheduler,
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = evaluate(model, val_dl, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                exp_dir / "checkpoints" / "best_language_model.pt",
            )

        if epoch % 5 == 0 or epoch == num_epochs:
            test_loss, test_acc = evaluate(model, test_dl, device)
            print(
                f"Epoch {epoch:03d}/{num_epochs} | Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% "
                f"| Test Acc: {test_acc*100:.2f}%"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{num_epochs} | Train Loss: {train_loss:.4f} "
                f"| Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
            )

        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_acc": val_acc,
            }
            torch.save(
                checkpoint,
                exp_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt",
            )
            save_training_log(
                train_losses,
                train_accs,
                val_losses,
                val_accs,
                path=exp_dir / "training_log.pkl",
            )

    torch.save(model.state_dict(), exp_dir / "final_language_model.pt")
    save_training_log(
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        path=exp_dir / "training_log.pkl",
    )
    print(
        f"\nTraining complete! Best validation accuracy: {best_val_acc*100:.2f}% "
        f"| Artifacts in {exp_dir}"
    )


if __name__ == "__main__":
    main()

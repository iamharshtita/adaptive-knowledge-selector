"""
Phase 1: Supervised Pre-training for Adaptive Knowledge Selector

Trains the DQN using labeled data with cross-entropy loss (imitation learning).
The model learns to predict the best source based on expert-labeled queries.

This provides a warm start before Phase 2 RL fine-tuning.

Usage:
    python scripts/pretrain_supervised.py
    python scripts/pretrain_supervised.py --epochs 30 --batch-size 32
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from models.adaptive_selector import AdaptiveSelector


class QueryDataset(Dataset):
    """Dataset for supervised pre-training."""

    def __init__(self, queries, labels, embeddings):
        """
        Args:
            queries: List of query strings
            labels: List of source labels (as indices)
            embeddings: Pre-computed query embeddings
        """
        self.queries = queries
        self.labels = labels
        self.embeddings = embeddings

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            'embedding': torch.FloatTensor(self.embeddings[idx]),
            'label': torch.LongTensor([self.labels[idx]])[0]
        }


def load_training_data(dataset_path: str):
    """Load and parse the training dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    queries = []
    best_sources = []

    for item in data:
        queries.append(item['query'])
        best_sources.append(item['best_source'])

    return queries, best_sources


def source_to_index(source_name: str) -> int:
    """Convert source name to index."""
    mapping = {
        "KnowledgeGraphSource": 0,
        "ToolAPISource": 1,
        "LLMSource": 2,
        "PDFKnowledgeSource": 3,
    }
    return mapping.get(source_name, -1)


def index_to_source(idx: int) -> str:
    """Convert index to source name."""
    mapping = {
        0: "KnowledgeGraphSource",
        1: "ToolAPISource",
        2: "LLMSource",
        3: "PDFKnowledgeSource",
    }
    return mapping.get(idx, "Unknown")


def evaluate(model, dataloader, device):
    """Evaluate model accuracy on a dataset."""
    model.policy_net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            outputs = model.policy_net(embeddings)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    model.policy_net.train()
    return correct / total if total > 0 else 0.0


def train_supervised(args):
    """Main training loop for supervised pre-training."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load training data
    print("\nLoading training dataset...")
    queries, best_sources = load_training_data(args.dataset)
    print(f"Loaded {len(queries)} queries")

    # Convert source names to indices
    labels = [source_to_index(src) for src in best_sources]

    # Check for invalid labels
    if -1 in labels:
        invalid_count = labels.count(-1)
        print(f"Warning: {invalid_count} queries have invalid source labels")
        # Filter out invalid entries
        valid_indices = [i for i, lbl in enumerate(labels) if lbl != -1]
        queries = [queries[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        print(f"Training on {len(queries)} valid queries")

    # Distribution of sources
    from collections import Counter
    label_dist = Counter(labels)
    print("\nSource distribution:")
    for idx, count in sorted(label_dist.items()):
        src_name = index_to_source(idx)
        print(f"  {src_name:<30s}: {count:>3d} ({100*count/len(labels):.1f}%)")

    # Encode queries
    print("\nEncoding queries with sentence transformer...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(queries, show_progress_bar=True, batch_size=64)
    print(f"Embeddings shape: {embeddings.shape}")

    # Train/validation split
    train_queries, val_queries, train_labels, val_labels, train_emb, val_emb = train_test_split(
        queries, labels, embeddings, test_size=0.15, random_state=42, stratify=labels
    )

    print(f"\nTrain set: {len(train_queries)} queries")
    print(f"Val set:   {len(val_queries)} queries")

    # Create datasets and dataloaders
    train_dataset = QueryDataset(train_queries, train_labels, train_emb)
    val_dataset = QueryDataset(val_queries, val_labels, val_emb)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print("\nInitializing model...")
    model = AdaptiveSelector(input_dim=384, num_sources=4, lr=args.lr)
    model.policy_net.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.policy_net.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    print("\n" + "=" * 70)
    print("  PHASE 1: SUPERVISED PRE-TRAINING")
    print("=" * 70)
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print("=" * 70 + "\n")

    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs': []
    }

    for epoch in range(1, args.epochs + 1):
        model.policy_net.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress_bar:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model.policy_net(embeddings)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy_net.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{epoch_correct/epoch_total:.3f}"
            })

        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total

        # Validation
        val_acc = evaluate(model, val_loader, device)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Log
        print(f"\n  Epoch {epoch:>3d}/{args.epochs}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Train Acc: {train_acc:.3f}  |  "
              f"Val Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.output)
            print(f"  New best! Saved to {args.output}")

        # Track history
        training_history['train_loss'].append(avg_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['epochs'].append(epoch)

        # Early stopping check
        if val_acc >= 0.95:
            print(f"\n  Reached {val_acc:.1%} validation accuracy. Stopping early.")
            break

    # Load best model for final evaluation
    model.load(args.output)

    # Final evaluation
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)

    final_train_acc = evaluate(model, train_loader, device)
    final_val_acc = evaluate(model, val_loader, device)

    print(f"\n  Final Training Accuracy:   {final_train_acc:.3f} ({final_train_acc*100:.1f}%)")
    print(f"  Final Validation Accuracy: {final_val_acc:.3f} ({final_val_acc*100:.1f}%)")
    print(f"  Best Validation Accuracy:  {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")

    # Per-source accuracy
    print("\n  Per-source validation accuracy:")
    model.policy_net.eval()
    source_correct = {i: 0 for i in range(4)}
    source_total = {i: 0 for i in range(4)}

    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            outputs = model.policy_net(embeddings)
            predictions = outputs.argmax(dim=1)

            for pred, label in zip(predictions, labels):
                label_val = label.item()
                source_total[label_val] += 1
                if pred == label:
                    source_correct[label_val] += 1

    for idx in range(4):
        src_name = index_to_source(idx)
        if source_total[idx] > 0:
            acc = source_correct[idx] / source_total[idx]
            print(f"    {src_name:<30s}: {acc:.3f} ({acc*100:.1f}%)  "
                  f"[{source_correct[idx]}/{source_total[idx]}]")

    # Test on sample queries
    print("\n  Sample predictions:")
    test_queries = [
        "What drugs interact with aspirin?",
        "Calculate BMI for 70 kg 1.75 m",
        "How does insulin regulate blood sugar?",
        "Latest research on statins",
        "KRR paper on ontologies"
    ]
    expected_sources = [
        "ToolAPISource",  # or KnowledgeGraphSource
        "ToolAPISource",
        "LLMSource",
        "ToolAPISource",  # or PDFKnowledgeSource
        "PDFKnowledgeSource"
    ]

    test_embeddings = encoder.encode(test_queries)
    model.policy_net.eval()

    with torch.no_grad():
        for i, (query, expected) in enumerate(zip(test_queries, expected_sources)):
            emb = torch.FloatTensor(test_embeddings[i]).unsqueeze(0).to(device)
            output = model.policy_net(emb)
            probs = torch.softmax(output, dim=1).squeeze()
            pred_idx = output.argmax().item()
            pred_source = index_to_source(pred_idx)

            print(f"\n    {i+1}. \"{query[:50]}...\"")
            print(f"       Predicted: {pred_source}  (expected: {expected})")
            print(f"       Confidence: {probs[pred_idx]:.2%}")

    # Save training history
    history_path = args.output.replace('.pth', '_pretrain_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n  Model saved to: {args.output}")
    print(f"  Training history saved to: {history_path}")
    print("=" * 70 + "\n")

    print("\nNext step: Run Phase 2 RL fine-tuning:")
    print(f"  python scripts/train_rl_agent.py --episodes 200 --pretrained {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Supervised Pre-training")
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/training_dataset_600.json',
        help='Path to training dataset with labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/rl_selector/pretrained_dqn.pth',
        help='Path to save pre-trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Number of training epochs (default: 25)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    args = parser.parse_args()
    train_supervised(args)
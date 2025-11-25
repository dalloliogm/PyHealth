#!/usr/bin/env python3
"""
Time-Series Modeling with VAE - Python Script Version

This script demonstrates how to use the enhanced VAE model for time-series data analysis and generation.
Unlike image VAEs that work with spatial patterns, time-series VAEs model sequential patterns in medical data.

Key Concepts:
- Sequence Encoding: RNN-based encoder captures temporal dependencies
- Latent Representation: Compressed representation of patient trajectories
- Sequence Generation: RNN decoder reconstructs realistic medical sequences

Applications:
- Patient trajectory modeling and generation
- Medical sequence anomaly detection
- Synthetic data generation for rare conditions
- Treatment pattern analysis
"""

from pyhealth.datasets import split_by_visit, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.models import VAE
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import InHospitalMortalityMIMIC4

import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Starting Time-Series VAE Modeling Script")
    print("\nSetup Instructions:")
    print("1. Download MIMIC4 demo data from: https://physionet.org/files/mimic-iv-demo/2.2/")
    print("2. Create a 'data/mimic4_demo' directory in your project root")
    print("3. Extract the downloaded files into 'data/mimic4_demo/hosp/' subdirectory")
    print("4. Update the ehr_root path below if needed")

    # Load MIMIC4 demo dataset
    print("\nLoading MIMIC4 demo dataset...")
    ehr_root = "data/mimic4_demo"  # Update this path to your local MIMIC4 demo data

    dataset = MIMIC4Dataset(
        ehr_root=ehr_root,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
    )

    # Set task for time-series modeling
    task = InHospitalMortalityMIMIC4()
    ts_dataset = dataset.set_task(task, num_workers=2)

    print("MIMIC4 demo dataset loaded")
    print(f"Number of samples: {len(ts_dataset)}")
    print(f"Input features: {list(ts_dataset.input_schema.keys())}")
    print(f"Output features: {list(ts_dataset.output_schema.keys())}")

    # Create and Train Time-Series VAE
    print("\nCreating time-series VAE model...")

    ts_model = VAE(
        dataset=ts_dataset,
        feature_keys=["conditions", "procedures"],  # Sequence features from MIMIC4
        label_key="mortality",
        mode="binary",  # Binary classification for mortality prediction
        input_type="timeseries",  # Key parameter for time-series mode
        hidden_dim=64,  # Latent dimension for medical sequences
    )

    print("Time-series VAE created")
    print(f"Input type: {ts_model.input_type}")
    print(f"Has embedding model: {hasattr(ts_model, 'embedding_model')}")
    print(f"Has RNN encoder: {hasattr(ts_model, 'encoder_rnn')}")
    print(f"Latent dimension: {ts_model.hidden_dim}")

    # Prepare data for training
    train_dataloader = get_dataloader(ts_dataset, batch_size=32, shuffle=True)

    # Create trainer
    trainer = Trainer(
        model=ts_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        metrics=["kl_divergence", "pr_auc", "roc_auc"]
    )

    # Train the model (reduced epochs for demo)
    print("\nTraining time-series VAE...")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=train_dataloader,  # Using same data for demo
        epochs=10,
        monitor="kl_divergence",
        monitor_criterion="min",
        optimizer_params={"lr": 1e-4},
    )

    print("Training completed!")

    # Evaluate Reconstruction Performance
    print("\nEvaluating reconstruction performance...")
    eval_results = trainer.evaluate(train_dataloader)

    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")

    # Get reconstruction examples
    data_batch = next(iter(train_dataloader))
    with torch.no_grad():
        output = ts_model(**data_batch)

    print(f"\nReconstruction shape: {output['y_prob'].shape}")
    print(f"Original shape: {output['y_true'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")

    # Generate New Medical Sequences
    print("\nGenerating new medical sequences...")
    ts_model.eval()
    with torch.no_grad():
        # Sample random latent vectors
        latent_samples = torch.randn(3, ts_model.hidden_dim).to(ts_model.device)

        # Decode to get sequence representations
        generated_sequences = ts_model.decoder(latent_samples)

        print("Generated sequence representations:")
        print(f"Shape: {generated_sequences.shape}")
        print(f"Sample values: {generated_sequences[0, :5].cpu().numpy()}")

        # Convert embeddings to human-understandable medical codes
        # Find closest codes in embedding space

        # Get all code embeddings from the embedding model
        all_codes = list(ts_model.embedding_model.code_vocab.keys())
        code_embeddings = ts_model.embedding_model.embeddings.weight.data  # [vocab_size, embed_dim]

        print(f"\nConverting to medical codes for generated sequence 0:")

        # For each position in the sequence, find closest codes
        seq_embeds = generated_sequences[0]  # [seq_len, embed_dim]

        # Compute cosine similarity with all code embeddings
        similarities = torch.matmul(seq_embeds, code_embeddings.t())  # [seq_len, vocab_size]

        # Get top 3 most similar codes for each position
        top_k = 3
        top_similarities, top_indices = torch.topk(similarities, top_k, dim=1)

        for pos in range(min(5, seq_embeds.shape[0])):  # Show first 5 positions
            codes = [all_codes[idx] for idx in top_indices[pos].cpu().numpy()]
            sims = top_similarities[pos].cpu().numpy()
            print(f"Position {pos}: {codes} (similarities: {sims})")

        print("\nNote: These represent the most likely medical codes for the generated sequence.")
        print("In practice, you might use beam search or other decoding strategies for better results.")

    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
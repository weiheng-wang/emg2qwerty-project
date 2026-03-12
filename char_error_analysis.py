"""Per-character error analysis script.

Usage (on Colab):
    !python char_error_analysis.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --model rnn_ctc \
        --transforms log_spectrogram

Loads a trained model, runs inference on the test set, and prints:
  1. Most commonly substituted character pairs
  2. Most commonly deleted characters
  3. Most commonly inserted characters
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import Levenshtein
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from emg2qwerty import transforms as T
from emg2qwerty.data import LabelData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="rnn_ctc")
    parser.add_argument("--transforms", type=str, default="log_spectrogram")
    parser.add_argument("--decoder", type=str, default="ctc_greedy")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    # Load config via Hydra (checkpoint and dataset.root set separately
    # to avoid Hydra resolver issues when running outside @hydra.main)
    config_dir = str(Path(__file__).parent / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="base",
            overrides=[
                f"model={args.model}",
                f"transforms={args.transforms}",
                f"decoder={args.decoder}",
                "user=single_user",
                "train=false",
            ],
        )
    OmegaConf.update(config, "checkpoint", args.checkpoint)
    OmegaConf.update(config, "dataset.root", args.data_root)

    pl.seed_everything(config.seed, workers=True)

    # Build transforms
    def _build_transform(configs):
        return T.Compose([instantiate(cfg) for cfg in configs])

    # Build session paths
    def _full_session_paths(dataset):
        sessions = [s["session"] for s in dataset]
        return [Path(args.data_root) / f"{s}.hdf5" for s in sessions]

    # Instantiate module from checkpoint
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    module = module.load_from_checkpoint(
        args.checkpoint,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
    )
    module.eval()

    # Instantiate datamodule
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )
    datamodule.setup("test")

    # Collect predictions and targets
    substitutions = Counter()  # (pred_char, target_char) -> count
    deletions = Counter()      # target_char -> count
    insertions = Counter()     # pred_char -> count
    char_total = Counter()     # target_char -> total occurrences

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(datamodule.test_dataloader()):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]
            N = len(input_lengths)

            emissions = module.forward(inputs)
            T_diff = inputs.shape[0] - emissions.shape[0]
            emission_lengths = input_lengths - T_diff

            predictions = module.decoder.decode_batch(
                emissions=emissions.cpu().numpy(),
                emission_lengths=emission_lengths.cpu().numpy(),
            )

            for i in range(N):
                target = LabelData.from_labels(
                    targets[: target_lengths[i], i].numpy()
                )
                pred_text = predictions[i].text
                target_text = target.text

                # Count target character frequencies
                for c in target_text:
                    char_total[c] += 1

                # Get edit operations
                editops = Levenshtein.editops(pred_text, target_text)
                for op, src_pos, dst_pos in editops:
                    if op == "replace":
                        substitutions[(pred_text[src_pos], target_text[dst_pos])] += 1
                    elif op == "delete":
                        # "delete" in editops means char in pred but not in target (insertion error)
                        insertions[pred_text[src_pos]] += 1
                    elif op == "insert":
                        # "insert" in editops means char in target but not in pred (deletion error)
                        deletions[target_text[dst_pos]] += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Print results
    print("\n" + "=" * 60)
    print("PER-CHARACTER ERROR ANALYSIS")
    print("=" * 60)

    print(f"\nTop {args.top_k} Substitution Errors (predicted -> actual):")
    print("-" * 45)
    for (pred_c, tgt_c), count in substitutions.most_common(args.top_k):
        pred_display = repr(pred_c) if pred_c == " " else pred_c
        tgt_display = repr(tgt_c) if tgt_c == " " else tgt_c
        print(f"  {pred_display:>6} -> {tgt_display:<6}  count: {count}")

    print(f"\nTop {args.top_k} Deletion Errors (chars in target missing from prediction):")
    print("-" * 45)
    for char, count in deletions.most_common(args.top_k):
        char_display = repr(char) if char == " " else char
        total = char_total.get(char, 0)
        rate = count / total * 100 if total > 0 else 0
        print(f"  {char_display:>6}  count: {count:>5}  (of {total}, {rate:.1f}% deleted)")

    print(f"\nTop {args.top_k} Insertion Errors (extra chars in prediction):")
    print("-" * 45)
    for char, count in insertions.most_common(args.top_k):
        char_display = repr(char) if char == " " else char
        print(f"  {char_display:>6}  count: {count}")

    print(f"\nPer-Character Error Rate (top {args.top_k} hardest characters):")
    print("-" * 45)
    char_errors = Counter()
    for (pred_c, tgt_c), count in substitutions.items():
        char_errors[tgt_c] += count
    for char, count in deletions.items():
        char_errors[char] += count

    char_error_rates = {}
    for char, total in char_total.items():
        if total >= 10:  # only chars with enough samples
            error_count = char_errors.get(char, 0)
            char_error_rates[char] = (error_count / total * 100, error_count, total)

    sorted_rates = sorted(char_error_rates.items(), key=lambda x: x[1][0], reverse=True)
    for char, (rate, errors, total) in sorted_rates[:args.top_k]:
        char_display = repr(char) if char == " " else char
        print(f"  {char_display:>6}  error rate: {rate:5.1f}%  ({errors}/{total})")

    print(f"\nEasiest characters (lowest error rate):")
    print("-" * 45)
    for char, (rate, errors, total) in sorted_rates[-args.top_k:]:
        char_display = repr(char) if char == " " else char
        print(f"  {char_display:>6}  error rate: {rate:5.1f}%  ({errors}/{total})")


if __name__ == "__main__":
    main()

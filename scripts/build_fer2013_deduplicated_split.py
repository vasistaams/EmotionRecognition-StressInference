"""Audit FER2013 pixel duplicates and write a reproducible split manifest."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from data.fer2013_dataset import FER2013_FOLDER_TO_LABEL, FER2013_CLASSES


def image_hash(path: Path) -> str:
    pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise OSError(f"Could not decode image: {path}")
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype=np.int64).tobytes())
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def collect_samples(root: Path, split: str) -> List[Dict]:
    samples = []
    for folder_name, label in FER2013_FOLDER_TO_LABEL.items():
        class_dir = root / split / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            samples.append(
                {
                    "path": path.relative_to(root.parent).as_posix(),
                    "label": label,
                    "class": FER2013_CLASSES[label],
                    "split": split,
                    "sha256": image_hash(path),
                }
            )
    return samples


def groups_by_hash(samples: Iterable[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["sha256"]].append(sample)
    return groups


def duplicate_summary(samples: List[Dict]) -> Dict:
    groups = groups_by_hash(samples)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    return {
        "image_count": len(samples),
        "unique_hash_count": len(groups),
        "duplicate_hash_groups": len(duplicate_groups),
        "extra_duplicate_images": sum(len(group) - 1 for group in duplicate_groups),
        "conflicting_label_groups": sum(
            len({sample["label"] for sample in group}) > 1
            for group in duplicate_groups
        ),
    }


def pair_summary(left: List[Dict], right: List[Dict]) -> Dict:
    left_groups = groups_by_hash(left)
    right_groups = groups_by_hash(right)
    shared = set(left_groups) & set(right_groups)
    conflicting_labels = 0
    for digest in shared:
        labels = {
            sample["label"]
            for sample in left_groups[digest] + right_groups[digest]
        }
        conflicting_labels += len(labels) > 1
    return {
        "shared_hash_groups": len(shared),
        "left_images_in_shared_groups": sum(
            len(left_groups[digest]) for digest in shared
        ),
        "right_images_in_shared_groups": sum(
            len(right_groups[digest]) for digest in shared
        ),
        "conflicting_label_hash_groups": conflicting_labels,
    }


def original_seeded_split(
    train_samples: List[Dict], seed: int, validation_ratio: float
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(len(FER2013_CLASSES)):
        class_samples = [sample for sample in train_samples if sample["label"] == label]
        indices = rng.permutation(len(class_samples))
        validation_count = int(round(len(class_samples) * validation_ratio))
        validation_indices = set(indices[:validation_count].tolist())
        validation.extend(
            sample for index, sample in enumerate(class_samples)
            if index in validation_indices
        )
        train.extend(
            sample for index, sample in enumerate(class_samples)
            if index not in validation_indices
        )
    return train, validation


def cleaned_stratified_split(
    train_samples: List[Dict],
    test_samples: List[Dict],
    seed: int,
    validation_ratio: float,
) -> Tuple[List[Dict], List[Dict], Dict]:
    test_hashes = {sample["sha256"] for sample in test_samples}
    eligible = [sample for sample in train_samples if sample["sha256"] not in test_hashes]
    removed_test_overlap = len(train_samples) - len(eligible)

    by_hash = groups_by_hash(eligible)
    conflicting_hashes = {
        digest
        for digest, group in by_hash.items()
        if len({sample["label"] for sample in group}) > 1
    }
    conflicting_samples = sum(len(by_hash[digest]) for digest in conflicting_hashes)
    eligible = [
        sample for sample in eligible if sample["sha256"] not in conflicting_hashes
    ]

    deduplicated = {}
    for sample in sorted(eligible, key=lambda item: item["path"]):
        deduplicated.setdefault(sample["sha256"], sample)
    unique_samples = list(deduplicated.values())
    train, validation = original_seeded_split(
        unique_samples, seed=seed, validation_ratio=validation_ratio
    )
    return train, validation, {
        "removed_train_images_matching_test_hashes": removed_test_overlap,
        "removed_train_images_internally_conflicting_label_groups": conflicting_samples,
        "removed_train_duplicate_copies_after_test_filter": (
            len(eligible) - len(unique_samples)
        ),
        "unique_clean_training_pool_images": len(unique_samples),
    }


def write_csv(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("path", "label", "class", "sha256")
        )
        writer.writeheader()
        writer.writerows(
            {
                key: sample[key]
                for key in ("path", "label", "class", "sha256")
            }
            for sample in sorted(samples, key=lambda item: item["path"])
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/FER2013"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproducibility/fer2013_deduplicated_seed42"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("--validation-ratio must be between 0 and 1")

    root = args.root if args.root.is_absolute() else REPOSITORY_ROOT / args.root
    root = root.absolute()
    output = args.output.resolve()
    original_train_samples = collect_samples(root, "train")
    test_samples = collect_samples(root, "test")
    old_train, old_validation = original_seeded_split(
        original_train_samples, args.seed, args.validation_ratio
    )
    clean_train, clean_validation, cleaning = cleaned_stratified_split(
        original_train_samples, test_samples, args.seed, args.validation_ratio
    )
    private_test = [
        sample for sample in test_samples
        if Path(sample["path"]).name.startswith("PrivateTest_")
    ]
    public_test = [
        sample for sample in test_samples
        if Path(sample["path"]).name.startswith("PublicTest_")
    ]
    if len(private_test) + len(public_test) != len(test_samples):
        raise ValueError(
            "Test filenames do not all identify PublicTest or PrivateTest provenance"
        )

    for name, samples in (
        ("train", clean_train),
        ("validation", clean_validation),
        ("test_combined", test_samples),
        ("test_private", private_test),
        ("test_public", public_test),
    ):
        write_csv(output / f"{name}.csv", samples)

    try:
        root_reference = root.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        root_reference = "UNVERIFIED_EXTERNAL_DATASET_ROOT"
    audit = {
        "root": root_reference,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "hash_definition": (
            "SHA-256 of decoded grayscale pixel array shape followed by pixel bytes"
        ),
        "label_order": FER2013_CLASSES,
        "original_dataset": {
            "training_folder_before_holdout": duplicate_summary(
                original_train_samples
            ),
            "train": duplicate_summary(old_train),
            "validation": duplicate_summary(old_validation),
            "test_combined": duplicate_summary(test_samples),
            "train_validation": pair_summary(old_train, old_validation),
            "train_test": pair_summary(old_train, test_samples),
            "validation_test": pair_summary(old_validation, test_samples),
            "private_test": duplicate_summary(private_test),
            "public_test": duplicate_summary(public_test),
            "train_class_counts": dict(Counter(s["class"] for s in old_train)),
            "validation_class_counts": dict(Counter(s["class"] for s in old_validation)),
        },
        "cleaned_split": {
            **cleaning,
            "train": duplicate_summary(clean_train),
            "validation": duplicate_summary(clean_validation),
            "train_validation": pair_summary(clean_train, clean_validation),
            "train_test_combined": pair_summary(clean_train, test_samples),
            "validation_test_combined": pair_summary(clean_validation, test_samples),
            "train_class_counts": dict(Counter(s["class"] for s in clean_train)),
            "validation_class_counts": dict(
                Counter(s["class"] for s in clean_validation)
            ),
            "test_combined_count": len(test_samples),
            "test_private_count": len(private_test),
            "test_public_count": len(public_test),
        },
        "files": {
            name: f"{name}.csv"
            for name in (
                "train",
                "validation",
                "test_combined",
                "test_private",
                "test_public",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

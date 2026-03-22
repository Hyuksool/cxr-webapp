"""
MIMIC-CXR Fine-tuning Pipeline — DenseNet121 Transfer Learning

Fine-tunes a DenseNet121 model on MIMIC-CXR-JPG dataset for 14-label
multi-label CheXpert classification.

Pipeline:
    Step 1: download  — Download MIMIC-CXR-JPG sample subset from PhysioNet
    Step 2: prepare   — Build label CSV from CheXpert labels
    Step 3: train     — Fine-tune DenseNet121 with frozen early layers
    Step 4: evaluate  — Compute AUC per pathology

Dataset:
    MIMIC-CXR-JPG v2.0.0 (PhysioNet, credentialed access)
    - 227,827 studies, 377,110 images
    - 14 CheXpert labels
    - For local training: use --max-samples to limit (default 5000)

Usage:
    # Full pipeline (with PhysioNet credentials)
    python train_cxr_classifier.py --step all \\
        --username YOUR_USERNAME --password YOUR_PASSWORD

    # Individual steps
    python train_cxr_classifier.py --step download --username X --password Y
    python train_cxr_classifier.py --step prepare
    python train_cxr_classifier.py --step train --epochs 20
    python train_cxr_classifier.py --step evaluate

    # Quick local test with sample data (no download)
    python train_cxr_classifier.py --step train --epochs 5 --batch-size 8 --max-samples 500
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "mimic_data" / "mimic-cxr"
MODELS_DIR = PROJECT_ROOT / "models"

# ─── PhysioNet config ────────────────────────────────────────────────────────
PHYSIONET_BASE = "https://physionet.org/files"
MIMIC_CXR_JPG_SLUG = "mimic-cxr-jpg/2.0.0"

# ─── CheXpert 14 labels ──────────────────────────────────────────────────────
CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
NUM_LABELS = len(CHEXPERT_LABELS)  # 14


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download
# ─────────────────────────────────────────────────────────────────────────────

def step_download(username: str, password: str, max_studies: int = 500):
    """
    Download MIMIC-CXR-JPG metadata and image subset from PhysioNet.

    Downloads:
    - mimic-cxr-2.0.0-chexpert.csv.gz  — CheXpert labels for all studies
    - mimic-cxr-2.0.0-split.csv.gz     — official train/val/test split
    - mimic-cxr-2.0.0-metadata.csv.gz  — DICOM metadata
    - A subset of JPG images (first max_studies studies)

    Args:
        username: PhysioNet username (CITI-certified)
        password: PhysioNet password
        max_studies: Number of studies to download (default 500 for local use)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Download metadata CSVs ─────────────────────────────────────────
    metadata_files = [
        "mimic-cxr-2.0.0-chexpert.csv.gz",
        "mimic-cxr-2.0.0-split.csv.gz",
        "mimic-cxr-2.0.0-metadata.csv.gz",
    ]

    for fname in metadata_files:
        target = DATA_DIR / fname
        if target.exists():
            logger.info(f"Already exists: {target}")
            continue
        url = f"{PHYSIONET_BASE}/{MIMIC_CXR_JPG_SLUG}/{fname}"
        logger.info(f"Downloading: {url}")
        _wget_download(url, target, username, password)

    # ── 2. Download image subset ──────────────────────────────────────────
    # MIMIC-CXR-JPG images are organized as:
    #   files/p{subject_id[:2]}/p{subject_id}/s{study_id}/*.jpg

    import gzip

    chexpert_path = DATA_DIR / "mimic-cxr-2.0.0-chexpert.csv.gz"
    if not chexpert_path.exists():
        logger.error("CheXpert labels not downloaded yet.")
        return

    logger.info(f"Reading CheXpert labels to build download list...")
    studies_to_download = []
    with gzip.open(chexpert_path, "rt") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_studies:
                break
            studies_to_download.append({
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
            })

    logger.info(f"Will download {len(studies_to_download)} studies")

    # Download study directories
    images_dir = DATA_DIR / "files"
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for study in studies_to_download:
        sid = study["subject_id"]
        stid = study["study_id"]
        p_prefix = f"p{sid[:2]}"
        p_full = f"p{sid}"
        s_full = f"s{stid}"

        local_dir = images_dir / p_prefix / p_full / s_full
        if local_dir.exists() and list(local_dir.glob("*.jpg")):
            downloaded += 1
            continue

        url = f"{PHYSIONET_BASE}/{MIMIC_CXR_JPG_SLUG}/files/{p_prefix}/{p_full}/{s_full}/"
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            _wget_mirror_dir(url, local_dir, username, password)
            downloaded += 1
            if downloaded % 50 == 0:
                logger.info(f"  Downloaded {downloaded}/{len(studies_to_download)} studies")
        except Exception as e:
            logger.warning(f"  Failed to download {s_full}: {e}")

    logger.info(f"Download complete: {downloaded} studies in {images_dir}")


def _wget_download(url: str, target: Path, username: str, password: str):
    """Download a single file via wget with PhysioNet credentials."""
    cmd = [
        "wget", "-q", "--show-progress",
        f"--user={username}", f"--password={password}",
        "-O", str(target),
        url,
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"wget failed for {url}")
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {url}")


def _wget_mirror_dir(url: str, local_dir: Path, username: str, password: str):
    """Mirror a PhysioNet directory (list + download JPGs)."""
    cmd = [
        "wget", "-q", "-r", "-l", "1", "-nd", "-A", "*.jpg",
        f"--user={username}", f"--password={password}",
        "-P", str(local_dir),
        url,
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Prepare labels
# ─────────────────────────────────────────────────────────────────────────────

def step_prepare(max_samples: int = 0):
    """
    Build training dataset from CheXpert labels + downloaded images.

    Reads mimic-cxr-2.0.0-chexpert.csv.gz and matches to downloaded images.
    Handles uncertain labels (mapped to 0 per U-zeros policy).
    Saves train/val/test splits to JSON.

    Args:
        max_samples: Limit total samples (0 = unlimited)
    """
    import gzip

    chexpert_path = DATA_DIR / "mimic-cxr-2.0.0-chexpert.csv.gz"
    split_path = DATA_DIR / "mimic-cxr-2.0.0-split.csv.gz"
    images_dir = DATA_DIR / "files"

    if not chexpert_path.exists():
        logger.error("CheXpert labels not found. Run --step download first.")
        return

    # ── 1. Load CheXpert labels ───────────────────────────────────────────
    logger.info("Loading CheXpert labels...")
    labels_by_study = {}
    with gzip.open(chexpert_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            study_id = row["study_id"]
            label_vec = []
            for label in CHEXPERT_LABELS:
                val = row.get(label, "")
                if val == "1.0" or val == "1":
                    label_vec.append(1)
                else:
                    # Treat missing/uncertain (-1) as 0 (U-zeros policy)
                    label_vec.append(0)
            labels_by_study[study_id] = label_vec

    logger.info(f"Loaded labels for {len(labels_by_study)} studies")

    # ── 2. Load official train/val/test split ─────────────────────────────
    split_by_dicom = {}
    if split_path.exists():
        with gzip.open(split_path, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split_by_dicom[row["dicom_id"]] = row["split"]

    # ── 3. Match images to labels ─────────────────────────────────────────
    logger.info("Matching downloaded images to labels...")
    dataset = {"train": [], "validate": [], "test": []}
    total_images = 0
    no_label = 0
    no_image = 0

    if not images_dir.exists():
        logger.warning("Images directory not found. Creating synthetic dataset for testing.")
        _create_synthetic_dataset(max_samples or 100)
        return

    for jpg_path in sorted(images_dir.rglob("*.jpg")):
        if max_samples and total_images >= max_samples:
            break

        # Extract study_id from path: .../s{study_id}/CXR_dicom_id.jpg
        parts = jpg_path.parts
        study_dir = None
        for part in parts:
            if part.startswith("s") and part[1:].isdigit():
                study_dir = part
                break

        if study_dir is None:
            continue

        study_id = study_dir[1:]  # remove leading 's'
        dicom_id = jpg_path.stem

        if study_id not in labels_by_study:
            no_label += 1
            continue

        split = split_by_dicom.get(dicom_id, "train")

        entry = {
            "image_path": str(jpg_path.relative_to(PROJECT_ROOT)),
            "study_id": study_id,
            "dicom_id": dicom_id,
            "labels": labels_by_study[study_id],
        }

        if split in ("train", "validate", "test"):
            dataset[split].append(entry)
        else:
            dataset["train"].append(entry)

        total_images += 1

    logger.info(f"Matched {total_images} images | No label: {no_label}")
    logger.info(f"Train: {len(dataset['train'])} | Val: {len(dataset['validate'])} | Test: {len(dataset['test'])}")

    # ── 4. Label statistics ───────────────────────────────────────────────
    all_records = dataset["train"] + dataset["validate"] + dataset["test"]
    label_counts = [0] * NUM_LABELS
    for rec in all_records:
        for i, v in enumerate(rec["labels"]):
            label_counts[i] += v

    logger.info("Label distribution:")
    for i, name in enumerate(CHEXPERT_LABELS):
        pct = label_counts[i] / max(len(all_records), 1) * 100
        logger.info(f"  {name:30s}: {label_counts[i]:5d} ({pct:.1f}%)")

    # ── 5. Save dataset ───────────────────────────────────────────────────
    output_path = DATA_DIR / "dataset.json"
    with open(output_path, "w") as f:
        json.dump({
            "total": total_images,
            "labels": CHEXPERT_LABELS,
            "label_counts": label_counts,
            "train": dataset["train"],
            "validate": dataset["validate"],
            "test": dataset["test"],
        }, f)
    logger.info(f"Saved dataset: {output_path}")


def _create_synthetic_dataset(n_samples: int):
    """Create a small synthetic dataset for testing the training pipeline."""
    import random
    logger.info(f"Creating synthetic dataset with {n_samples} samples...")

    # Create dummy image directory
    synth_dir = DATA_DIR / "synthetic_images"
    synth_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False

    records = []
    for i in range(n_samples):
        img_path = synth_dir / f"synth_{i:04d}.jpg"
        if has_pil and not img_path.exists():
            # Create a small synthetic grayscale image (224x224)
            arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            img = Image.fromarray(arr, mode="L")
            img.save(str(img_path))

        # Random multi-label vector (sparse, ~20% positive rate)
        labels = [1 if np.random.random() < 0.2 else 0 for _ in range(NUM_LABELS)]
        records.append({
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "study_id": f"synth_{i:04d}",
            "dicom_id": f"synth_{i:04d}",
            "labels": labels,
        })

    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    label_counts = [sum(r["labels"][i] for r in records) for i in range(NUM_LABELS)]

    output_path = DATA_DIR / "dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "total": n_samples,
            "labels": CHEXPERT_LABELS,
            "label_counts": label_counts,
            "train": train,
            "validate": val,
            "test": test,
        }, f)
    logger.info(f"Synthetic dataset saved: {output_path}")


class CXRDataset:
    """MIMIC-CXR image dataset with multi-label CheXpert labels."""

    def __init__(self, records: list, transform, base_dir: Path):
        self.records = records
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import torch
        from PIL import Image as PILImage
        rec = self.records[idx]
        img_path = self.base_dir / rec["image_path"]

        if img_path.exists():
            try:
                img = PILImage.open(img_path).convert("RGB")
            except Exception:
                img = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            img = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        img = self.transform(img)
        labels = torch.FloatTensor(rec["labels"])
        return img, labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Train
# ─────────────────────────────────────────────────────────────────────────────

def step_train(epochs: int = 20, batch_size: int = 16, lr: float = 1e-4,
               max_samples: int = 0, freeze_layers: int = 8):
    """
    Fine-tune DenseNet121 on MIMIC-CXR labels.

    Architecture:
    - Pretrained DenseNet121 (ImageNet weights via TorchXRayVision or torchvision)
    - Freeze first N dense blocks for transfer learning
    - Replace classifier head: Linear(1024, 14) + Sigmoid
    - Loss: BCEWithLogitsLoss with positive class weights
    - Optimizer: AdamW + CosineAnnealingLR
    - Augmentation: Random horizontal flip, rotation ±10°, crop

    Args:
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Initial learning rate
        max_samples: Limit training samples (0 = use all)
        freeze_layers: Number of DenseNet layers to freeze (0 = fine-tune all)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models

    try:
        from sklearn.metrics import roc_auc_score
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    # ── 1. Load dataset ───────────────────────────────────────────────────
    dataset_path = DATA_DIR / "dataset.json"
    if not dataset_path.exists():
        logger.warning("Dataset not found. Creating synthetic dataset...")
        _create_synthetic_dataset(max_samples or 200)

    with open(dataset_path) as f:
        ds = json.load(f)

    train_records = ds["train"]
    val_records = ds["validate"]
    label_counts = ds.get("label_counts", [1] * NUM_LABELS)
    total = len(train_records) + len(val_records) + len(ds["test"])

    if max_samples:
        train_records = train_records[:max_samples]
        val_records = val_records[:max_samples // 5]

    logger.info(f"Train: {len(train_records)} | Val: {len(val_records)}")

    # ── 3. Transforms ─────────────────────────────────────────────────────
    # CXR standard: normalize to ImageNet stats (DenseNet121 pretrained)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # ── 4. Data loaders ───────────────────────────────────────────────────
    train_dataset = CXRDataset(train_records, train_transform, PROJECT_ROOT)
    val_dataset = CXRDataset(val_records, val_transform, PROJECT_ROOT)
    # num_workers=0 for macOS MPS/spawn compatibility
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    # ── 5. Model ──────────────────────────────────────────────────────────
    # Try TorchXRayVision pretrained (best CXR initialization)
    model = None
    try:
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_LABELS)
        logger.info("Using TorchXRayVision DenseNet121 (pretrained on CXR)")
    except (ImportError, Exception) as e:
        logger.info(f"TorchXRayVision not available ({e}), falling back to torchvision")

    if model is None:
        # Try downloading pretrained weights; fall back to random init on SSL error
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            model = models.densenet121(weights="IMAGENET1K_V1")
            logger.info("Using torchvision DenseNet121 (ImageNet pretrained)")
        except Exception as e:
            logger.warning(f"Pretrained weights unavailable ({e}), using random init")
            model = models.densenet121(weights=None)
            logger.info("Using torchvision DenseNet121 (random init)")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_LABELS)

    # Selective layer freezing for transfer learning
    if freeze_layers > 0:
        frozen = 0
        for name, param in model.named_parameters():
            if frozen < freeze_layers and "classifier" not in name:
                param.requires_grad = False
                frozen += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Frozen {freeze_layers} layers | Trainable params: {trainable:,}")

    # Device selection: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Training on: {device}")
    model = model.to(device)

    # ── 6. Loss, optimizer, scheduler ────────────────────────────────────
    # Positive class weights for imbalanced labels
    pos_weights = []
    for count in label_counts:
        weight = (total - count) / max(count, 1)
        pos_weights.append(min(weight, 15.0))  # cap at 15x
    pos_weight = torch.FloatTensor(pos_weights).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 7. Training loop ──────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_auc = 0.0
    history = []

    logger.info("=" * 60)
    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            # Handle TorchXRayVision output (may return dict or tensor)
            if isinstance(logits, dict):
                logits = logits.get("logit", list(logits.values())[0])
            if logits.shape[-1] != NUM_LABELS:
                logits = logits[:, :NUM_LABELS]

            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_imgs)
                if isinstance(logits, dict):
                    logits = logits.get("logit", list(logits.values())[0])
                if logits.shape[-1] != NUM_LABELS:
                    logits = logits[:, :NUM_LABELS]
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(batch_labels.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step()

        # Compute mean AUC
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        mean_auc = 0.0
        if has_sklearn and len(val_records) > 0:
            aucs = []
            for i in range(NUM_LABELS):
                if all_labels[:, i].sum() > 0 and all_labels[:, i].sum() < len(all_labels):
                    try:
                        aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
                    except Exception:
                        pass
            mean_auc = np.mean(aucs) if aucs else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"AUC: {mean_auc:.4f} | LR: {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_auc": mean_auc,
        })

        # Save best model (by AUC, fallback to loss)
        improved = mean_auc > best_val_auc if mean_auc > 0 else val_loss < best_val_loss
        if improved:
            best_val_auc = mean_auc
            best_val_loss = val_loss
            best_path = MODELS_DIR / "cxr_densenet121_finetuned.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "mean_auc": mean_auc,
                "labels": CHEXPERT_LABELS,
                "num_labels": NUM_LABELS,
            }, str(best_path))
            logger.info(f"  Saved best model: {best_path}")

    # Save training history
    history_path = MODELS_DIR / "cxr_training_history.json"
    with open(history_path, "w") as f:
        json.dump({"epochs": epochs, "history": history}, f, indent=2)
    logger.info(f"Training complete. Best AUC: {best_val_auc:.4f}")
    logger.info(f"Model saved: {MODELS_DIR / 'cxr_densenet121_finetuned.pth'}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def step_evaluate():
    """
    Evaluate the trained model on the test set.

    Reports per-label AUC, F1, and overall metrics.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms, models

    try:
        from sklearn.metrics import roc_auc_score, f1_score, classification_report
        has_sklearn = True
    except ImportError:
        logger.error("scikit-learn required for evaluation. pip install scikit-learn")
        return

    # Load dataset
    dataset_path = DATA_DIR / "dataset.json"
    if not dataset_path.exists():
        logger.error("Dataset not found. Run --step prepare first.")
        return

    with open(dataset_path) as f:
        ds = json.load(f)

    test_records = ds["test"]
    if not test_records:
        logger.warning("No test records found. Using validation set.")
        test_records = ds["validate"]

    logger.info(f"Evaluating on {len(test_records)} test samples")

    # Load model
    model_path = MODELS_DIR / "cxr_densenet121_finetuned.pth"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    checkpoint = torch.load(str(model_path), map_location="cpu")

    try:
        import torchxrayvision as xrv
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        import torch.nn as nn
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_LABELS)
    except ImportError:
        model = models.densenet121()
        import torch.nn as nn
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_LABELS)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)

    # Evaluate
    from PIL import Image
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    all_probs, all_labels = [], []
    for rec in test_records:
        img_path = PROJECT_ROOT / rec["image_path"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        img_t = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_t)
            if isinstance(logits, dict):
                logits = logits.get("logit", list(logits.values())[0])
            if logits.shape[-1] != NUM_LABELS:
                logits = logits[:, :NUM_LABELS]
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        all_probs.append(probs)
        all_labels.append(rec["labels"])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Per-label AUC
    logger.info("\n=== Per-Label AUC ===")
    aucs = {}
    for i, label in enumerate(CHEXPERT_LABELS):
        if all_labels[:, i].sum() > 0:
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                aucs[label] = auc
                logger.info(f"  {label:30s}: AUC = {auc:.4f}")
            except Exception as e:
                logger.warning(f"  {label}: AUC failed ({e})")
        else:
            logger.info(f"  {label:30s}: No positive samples in test set")

    mean_auc = np.mean(list(aucs.values())) if aucs else 0.0
    logger.info(f"\nMean AUC: {mean_auc:.4f}")

    # Save evaluation results
    results_path = MODELS_DIR / "cxr_eval_results.json"
    with open(results_path, "w") as f:
        json.dump({"mean_auc": mean_auc, "per_label_auc": aucs}, f, indent=2)
    logger.info(f"Saved evaluation results: {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-CXR Fine-tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", required=True,
        choices=["download", "prepare", "train", "evaluate", "all", "quick-test"],
        help="Pipeline step to run",
    )
    parser.add_argument("--username", help="PhysioNet username")
    parser.add_argument("--password", help="PhysioNet password")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit samples for testing (0=all)")
    parser.add_argument("--max-studies", type=int, default=500,
                        help="Max studies to download (default: 500)")
    parser.add_argument("--freeze-layers", type=int, default=8,
                        help="DenseNet layers to freeze (default: 8)")

    args = parser.parse_args()

    if args.step == "quick-test":
        # Quick test: synthetic data + 3 epochs, no download needed
        logger.info("Quick test mode: synthetic data, 3 epochs")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        _create_synthetic_dataset(200)
        step_train(epochs=3, batch_size=8, lr=1e-3, max_samples=200, freeze_layers=0)
        step_evaluate()
        return

    if args.step in ("download", "all"):
        if not args.username or not args.password:
            logger.error("--username and --password required for download step")
            logger.info("PhysioNet credentials: https://physionet.org/register/")
            logger.info("MIMIC-CXR requires credentialed access (CITI training)")
            sys.exit(1)
        step_download(args.username, args.password, args.max_studies)

    if args.step in ("prepare", "all"):
        step_prepare(args.max_samples)

    if args.step in ("train", "all"):
        step_train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_samples=args.max_samples,
            freeze_layers=args.freeze_layers,
        )

    if args.step in ("evaluate", "all"):
        step_evaluate()


if __name__ == "__main__":
    main()

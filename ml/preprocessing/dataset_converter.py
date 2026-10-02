import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ml.config.settings import (
    CLASS_ALIASES,
    CLASSES,
    PROCESSED_DATASET_ROOT,
    RAW_DATASET_ROOT,
)


def create_class_directories(output_root: Path = PROCESSED_DATASET_ROOT) -> None:
    for split in ("train", "validation"):
        for class_name in CLASSES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)


def convert_dataset(
    image_dir: Path,
    annotation_dir: Path,
    split: str,
    output_root: Path = PROCESSED_DATASET_ROOT,
) -> int:
    """Convert one-object XML annotations into Keras class folders."""
    copied = 0
    for xml_path in sorted(annotation_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        object_node = root.find("object")
        class_node = object_node.find("name") if object_node is not None else None
        raw_class_name = class_node.text.strip() if class_node is not None and class_node.text else ""
        class_name = CLASS_ALIASES.get(raw_class_name, raw_class_name)

        if class_name not in CLASSES:
            raise ValueError(
                f"Unsupported or missing class in {xml_path}: {raw_class_name!r}"
            )

        image_matches = [
            candidate
            for candidate in image_dir.rglob(f"{xml_path.stem}.*")
            if candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not image_matches:
            continue
        image_path = image_matches[0]

        destination = output_root / split / class_name / image_path.name
        shutil.copy2(image_path, destination)
        copied += 1

    return copied


def prepare_dataset(
    raw_root: Path = RAW_DATASET_ROOT,
    output_root: Path = PROCESSED_DATASET_ROOT,
) -> dict[str, int]:
    create_class_directories(output_root)
    counts = {}
    for split in ("train", "validation"):
        counts[split] = convert_dataset(
            raw_root / split / "images",
            raw_root / split / "annotations",
            split,
            output_root,
        )
    return counts


if __name__ == "__main__":
    print(prepare_dataset())

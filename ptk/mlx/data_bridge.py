"""Bridge PTK TableDict datasets to mlx-lm local JSONL layout."""

from __future__ import annotations

import json
from pathlib import Path

from ptk.data.table import Table, TableDict
from ptk.exceptions import ValidationError


def _text_column(table: Table) -> str:
    if "text" in table.column_names:
        return "text"
    if "prompt" in table.column_names:
        return "prompt"
    raise ValidationError(
        "MLX data bridge requires a 'text' (or 'prompt') column in the training table",
        field_path="data",
        details={"columns": list(table.column_names)},
    )


def write_mlx_dataset(
    dataset: TableDict,
    out_dir: Path,
    *,
    batch_size: int = 1,
) -> Path:
    """Write train.jsonl and optional valid.jsonl for mlx-lm.

    mlx-lm expects filenames ``train.jsonl`` / ``valid.jsonl`` (not validation).
    """
    if "train" not in dataset:
        raise ValidationError(
            "MLX training requires a train split",
            field_path="data",
        )

    train = dataset["train"]
    if len(train) < batch_size:
        raise ValidationError(
            f"MLX training requires at least batch_size={batch_size} train examples, "
            f"but only {len(train)} are available. Reduce training.batch_size or add data.",
            field_path="training.batch_size",
            details={"train_rows": len(train), "batch_size": batch_size},
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_key = _text_column(train)
    _write_jsonl(out_dir / "train.jsonl", train, text_key)

    if "validation" in dataset and len(dataset["validation"]) > 0:
        val = dataset["validation"]
        val_key = _text_column(val) if "text" in val.column_names or "prompt" in val.column_names else text_key
        # Prefer text column when present
        if "text" in val.column_names:
            val_key = "text"
        elif "prompt" in val.column_names:
            val_key = "prompt"
        _write_jsonl(out_dir / "valid.jsonl", val, val_key)

    return out_dir


def _write_jsonl(path: Path, table: Table, text_key: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i in range(len(table)):
            row = table[i]
            text = str(row.get(text_key, "")).strip()
            if not text:
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

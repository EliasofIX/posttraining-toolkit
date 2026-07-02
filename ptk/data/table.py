"""In-memory tabular dataset (replacement for HuggingFace datasets in ptk code)."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


class Table:
    """Column-oriented in-memory table."""

    def __init__(self, columns: list[str], rows: list[dict[str, Any]]) -> None:
        self._columns = list(columns)
        self._rows = rows

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> Table:
        if not data:
            return cls([], [])
        columns = list(data.keys())
        lengths = {col: len(data[col]) for col in columns}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(f"All columns must have equal length, got: {lengths}")
        length = next(iter(unique_lengths))
        rows = [{col: data[col][i] for col in columns} for i in range(length)]
        return cls(columns, rows)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> Table:
        if not records:
            return cls([], [])
        columns = list(records[0].keys())
        return cls(columns, [dict(record) for record in records])

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    def get(self, split_name: str) -> Table:
        return self

    def map(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        remove_columns: list[str] | None = None,
    ) -> Table:
        new_rows = [fn(dict(row)) for row in self._rows]
        remove = set(remove_columns or [])
        columns = [c for c in self._infer_columns(new_rows) if c not in remove]
        normalized = [{col: row.get(col) for col in columns} for row in new_rows]
        return Table(columns, normalized)

    def filter(self, fn: Callable[[dict[str, Any]], bool]) -> Table:
        kept = [row for row in self._rows if fn(dict(row))]
        columns = self._columns or self._infer_columns(kept)
        return Table(columns, kept)

    def select(self, indices: list[int]) -> Table:
        kept = [self._rows[i] for i in indices]
        return Table(self._columns, kept)

    def rename_columns(self, mapping: dict[str, str]) -> Table:
        columns = [mapping.get(col, col) for col in self._columns]
        rows = [{mapping.get(k, k): v for k, v in row.items()} for row in self._rows]
        return Table(columns, rows)

    def train_test_split(
        self,
        *,
        test_size: float,
        seed: int = 42,
    ) -> dict[str, Table]:
        indices = list(range(len(self._rows)))
        if len(indices) <= 1:
            return {"train": self.select(indices), "test": self.select([])}

        rng = random.Random(seed)
        rng.shuffle(indices)
        test_count = max(1, round(len(indices) * test_size))
        if test_count >= len(indices):
            test_count = len(indices) - 1
        test_idx = set(indices[:test_count])
        train_indices = [i for i in indices if i not in test_idx]
        test_indices = [i for i in indices if i in test_idx]
        return {
            "train": self.select(train_indices),
            "test": self.select(test_indices),
        }

    def save_jsonl(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "data.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in self._rows:
                handle.write(json.dumps(row, default=str) + "\n")

    @classmethod
    def load_jsonl(cls, directory: str | Path) -> Table:
        directory = Path(directory)
        path = directory / "data.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls.from_records(rows)

    def _infer_columns(self, rows: list[dict[str, Any]]) -> list[str]:
        if self._columns:
            return self._columns
        if not rows:
            return []
        return list(rows[0].keys())


class TableDict(dict[str, Table]):
    """Named table splits (train / validation)."""

    def get(self, key: str, default: Table | None = None) -> Table | None:  # type: ignore[override]
        return super().get(key, default)

    @property
    def column_names(self) -> list[str]:
        first = next(iter(self.values()), None)
        return first.column_names if first else []

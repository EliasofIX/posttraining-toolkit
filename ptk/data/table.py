"""In-memory tabular dataset (replacement for HuggingFace datasets in ptk code)."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


class Table:
    """Column-oriented in-memory table."""

    def __init__(self, columns: list[str], data: dict[str, list[Any]]) -> None:
        self._columns = list(columns)
        self._data = {col: data[col] for col in columns}

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> Table:
        if not data:
            return cls([], {})
        columns = list(data.keys())
        lengths = {col: len(data[col]) for col in columns}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(f"All columns must have equal length, got: {lengths}")
        return cls(columns, {col: list(data[col]) for col in columns})

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> Table:
        if not records:
            return cls([], {})
        columns = list(records[0].keys())
        data = {col: [record.get(col) for record in records] for col in columns}
        return cls(columns, data)

    def to_dict(self) -> dict[str, list[Any]]:
        """Return column-oriented data for zero-copy HF Dataset conversion."""
        return {col: self._data[col] for col in self._columns}

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def __len__(self) -> int:
        if not self._columns:
            return 0
        return len(self._data[self._columns[0]])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {col: self._data[col][index] for col in self._columns}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def get(self, split_name: str) -> Table:
        return self

    def map(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        remove_columns: list[str] | None = None,
    ) -> Table:
        new_rows = [fn(self[index]) for index in range(len(self))]
        remove = set(remove_columns or [])
        columns = [col for col in self._infer_columns(new_rows) if col not in remove]
        data = {col: [row.get(col) for row in new_rows] for col in columns}
        return Table(columns, data)

    def filter(self, fn: Callable[[dict[str, Any]], bool]) -> Table:
        kept_indices = [index for index in range(len(self)) if fn(self[index])]
        return self.select(kept_indices)

    def select(self, indices: list[int]) -> Table:
        data = {col: [self._data[col][index] for index in indices] for col in self._columns}
        return Table(self._columns, data)

    def rename_columns(self, mapping: dict[str, str]) -> Table:
        columns = [mapping.get(col, col) for col in self._columns]
        data = {
            mapping.get(col, col): list(self._data[col])
            for col in self._columns
        }
        return Table(columns, data)

    def train_test_split(
        self,
        *,
        test_size: float,
        seed: int = 42,
    ) -> dict[str, Table]:
        indices = list(range(len(self)))
        if len(indices) <= 1:
            return {"train": self.select(indices), "test": self.select([])}

        rng = random.Random(seed)
        rng.shuffle(indices)
        test_count = max(1, round(len(indices) * test_size))
        if test_count >= len(indices):
            test_count = len(indices) - 1
        test_idx = set(indices[:test_count])
        train_indices = [index for index in indices if index not in test_idx]
        test_indices = [index for index in indices if index in test_idx]
        return {
            "train": self.select(train_indices),
            "test": self.select(test_indices),
        }

    def save_jsonl(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "data.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in self:
                handle.write(json.dumps(row, default=str) + "\n")

    @classmethod
    def load_jsonl(cls, directory: str | Path) -> Table:
        directory = Path(directory)
        path = directory / "data.jsonl"
        columns: list[str] = []
        data: dict[str, list[Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not columns:
                    columns = list(row.keys())
                    data = {col: [] for col in columns}
                for col in columns:
                    data[col].append(row.get(col))
        return cls(columns, data)

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

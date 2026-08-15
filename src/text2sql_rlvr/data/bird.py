"""Locate and load a BIRD split.

BIRD ships each release with a slightly different directory name and, across
mini-dev / dev / train, slightly different JSON field names. Rather than hard-code
one layout and break on the next download, discovery searches for the known file
names under a root and reports precisely what it looked for when it fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Split name -> candidate question-file names, in preference order.
QUESTION_FILES: dict[str, tuple[str, ...]] = {
    "mini_dev": ("mini_dev_sqlite.json", "mini_dev.json"),
    "dev": ("dev.json",),
    "train": ("train.json",),
}

#: Split name -> candidate database directory names, in preference order.
DATABASE_DIRS: dict[str, tuple[str, ...]] = {
    "mini_dev": ("dev_databases", "mini_dev_databases"),
    "dev": ("dev_databases",),
    "train": ("train_databases",),
}

SPLITS = tuple(QUESTION_FILES)

_GOLD_KEYS = ("SQL", "sql", "query", "gold_sql")


@dataclass(frozen=True)
class BirdExample:
    """One BIRD question."""

    question_id: int
    db_id: str
    question: str
    evidence: str = ""
    gold_sql: str = ""
    difficulty: str | None = None


@dataclass(frozen=True)
class BirdSplit:
    """A resolved split: where its questions live and where its databases live."""

    name: str
    questions_path: Path
    databases_dir: Path

    def db_path(self, db_id: str) -> Path:
        return self.databases_dir / db_id / f"{db_id}.sqlite"

    def load(self) -> list[BirdExample]:
        return load_examples(self.questions_path)

    def missing_databases(self, examples: list[BirdExample]) -> list[str]:
        """Database ids referenced by ``examples`` whose SQLite file is absent."""
        seen: dict[str, bool] = {}
        for example in examples:
            if example.db_id not in seen:
                seen[example.db_id] = self.db_path(example.db_id).is_file()
        return sorted(db_id for db_id, present in seen.items() if not present)


def load_examples(path: str | Path) -> list[BirdExample]:
    """Parse a BIRD question file, tolerating the field-name drift between releases."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} should contain a JSON list, got {type(raw).__name__}")

    examples: list[BirdExample] = []
    for index, item in enumerate(raw):
        if "db_id" not in item or "question" not in item:
            raise ValueError(f"{path}[{index}] is missing db_id or question")
        gold = ""
        for key in _GOLD_KEYS:
            if item.get(key):
                gold = str(item[key]).strip()
                break
        examples.append(
            BirdExample(
                question_id=int(item.get("question_id", index)),
                db_id=str(item["db_id"]),
                question=str(item["question"]).strip(),
                evidence=str(item.get("evidence") or "").strip(),
                gold_sql=gold,
                difficulty=item.get("difficulty"),
            )
        )
    return examples


def _find_file(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        for candidate in sorted(root.rglob(name)):
            if candidate.is_file():
                return candidate
    return None


def _find_dir(root: Path, near: Path, names: tuple[str, ...]) -> Path | None:
    for base in (near, near.parent, root):
        for name in names:
            candidate = base / name
            if candidate.is_dir():
                return candidate
    for name in names:
        for candidate in sorted(root.rglob(name)):
            if candidate.is_dir():
                return candidate
    return None


def discover_split(root: str | Path, name: str) -> BirdSplit:
    """Find the question file and database directory for ``name`` under ``root``."""
    if name not in QUESTION_FILES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root.resolve()}")

    question_names = QUESTION_FILES[name]
    questions = _find_file(root, question_names)
    if questions is None:
        raise FileNotFoundError(
            f"no question file for split {name!r} under {root.resolve()}; "
            f"looked for {', '.join(question_names)}"
        )

    dir_names = DATABASE_DIRS[name]
    databases = _find_dir(root, questions.parent, dir_names)
    if databases is None:
        raise FileNotFoundError(
            f"no database directory for split {name!r} under {root.resolve()}; "
            f"looked for {', '.join(dir_names)}"
        )

    return BirdSplit(name=name, questions_path=questions, databases_dir=databases)

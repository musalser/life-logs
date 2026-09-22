"""Inspect and change which HTR model an author uses (model rollback).

The lifecycle already keeps every version and only one of them ACTIVE. This
script is the operational escape hatch for the cases the HTTP API does not
cover yet: rolling back to the default model after a fine-tune turned out to be
worse, or re-activating an earlier version.

Usage (from the ``backend`` directory)::

    .venv/bin/python scripts/set_active_htr_model.py --author-id 1 --list
    .venv/bin/python scripts/set_active_htr_model.py --author-id 1 --default
    .venv/bin/python scripts/set_active_htr_model.py --author-id 1 --version 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.htr.domain.entities import ModelRef  # noqa: E402
from app.htr.infrastructure.model_repository import SqlAlchemyModelRepository  # noqa: E402
from app.htr.infrastructure.storage import HTRStorage  # noqa: E402


def build_repo(db):
    default = ModelRef(
        id=settings.htr_default_model_id, path=settings.htr_default_model_path
    )
    return SqlAlchemyModelRepository(db, HTRStorage(settings.htr_storage_dir), default)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-id", type=int, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="print all versions")
    group.add_argument("--default", action="store_true", help="roll back to the default model")
    group.add_argument("--version", type=int, help="activate this version number")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        repo = build_repo(db)
        versions = repo.list_versions(args.author_id)
        if args.list or not (args.default or args.version):
            print(f"default model: {repo.get_default_model().path}")
            for v in versions:
                metrics = v.metrics or {}
                print(
                    f"  v{v.version} id={v.id} status={v.status.value} cer={metrics.get('cer')} "
                    f"base={v.base_model_id} file={v.file_path}"
                )
            active = repo.get_active_model(args.author_id)
            print(f"active: {'default' if active is None else f'v{active.version}'}")
            return 0

        if args.default:
            repo.clear_active_model(args.author_id)
            print(f"author {args.author_id}: rolled back to the default model")
            return 0

        target = next((v for v in versions if v.version == args.version), None)
        if target is None:
            raise SystemExit(f"author {args.author_id} has no version {args.version}")
        if not Path(target.file_path).is_file():
            raise SystemExit(f"version {args.version} has no model file: {target.file_path}")
        repo.activate_model(args.author_id, target.id)
        print(f"author {args.author_id}: activated v{target.version} ({target.file_path})")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Infrastructure: the on-disk ``dataset.dcp`` directory (Parquet tables plus a JSON manifest)."""

from datacopilot.store.dataset_store import DatasetStore, StoreLimits, ensure_outside

__all__ = ["DatasetStore", "StoreLimits", "ensure_outside"]

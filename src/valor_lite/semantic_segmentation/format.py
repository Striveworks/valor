from pathlib import Path


class PathFormatter:
    @staticmethod
    def _generate_cache_path(path: str | Path) -> Path:
        return Path(path) / "counts"

    @staticmethod
    def _generate_metadata_path(path: str | Path) -> Path:
        return Path(path) / "metadata.json"

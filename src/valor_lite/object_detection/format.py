from pathlib import Path


class PathFormatter:
    @staticmethod
    def _generate_detailed_cache_path(path: str | Path) -> Path:
        return Path(path) / "detailed"

    @staticmethod
    def _generate_ranked_cache_path(path: str | Path) -> Path:
        return Path(path) / "ranked"

    @staticmethod
    def _generate_metadata_path(path: str | Path) -> Path:
        return Path(path) / "metadata.json"

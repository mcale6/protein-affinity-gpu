from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator


SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}


@contextmanager
def data_path(filename: str) -> Iterator[Path]:
    """Yield a concrete filesystem path for a packaged data resource."""
    resource = files("protein_affinity_gpu").joinpath("data", filename)
    with as_file(resource) as resolved_path:
        yield resolved_path


def read_text_resource(filename: str) -> str:
    """Read a packaged data file as text."""
    with data_path(filename) as resolved_path:
        return resolved_path.read_text()


def collect_structure_files(input_path: Path) -> list[Path]:
    """Collect supported structure files from a file or directory."""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
            raise ValueError(f"Unsupported structure file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    structure_files = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_STRUCTURE_SUFFIXES
    )
    if not structure_files:
        raise ValueError(f"No structure files found in directory: {input_path}")
    return structure_files


def format_duration(seconds: float) -> str:
    """Format a duration as a short human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes = int(seconds // 60)
    remainder = seconds % 60
    return f"{minutes}m {remainder:.2f}s"

from pathlib import Path


def read_cuda_source(cuda_dir: Path, name: str) -> str:
    return (cuda_dir / name).read_text(encoding="utf-8")


def cuda_compile_options(
    cuda_dir: Path,
    *options: str,
) -> tuple[str, ...]:
    return (
        *options,
        f"--include-path={cuda_dir.resolve()}",
    )

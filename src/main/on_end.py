import os
from pathlib import Path


def rename_checkpoint_file(checkpoint_dir: Path | str) -> None:
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path_list = list(checkpoint_dir.glob("**/*.ckpt"))
    for checkpoint_path in checkpoint_path_list:
        dest_path = Path(
            str(checkpoint_path).replace(
                checkpoint_path.stem, checkpoint_path.stem.replace("=", ":")
            )
        )
        os.rename(str(checkpoint_path), str(dest_path))

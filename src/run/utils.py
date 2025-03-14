from pathlib import Path

from src.main.on_end import rename_checkpoint_file


def get_last_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    rename_checkpoint_file(checkpoint_dir)
    checkpoint_path_list = list(checkpoint_dir.glob("**/*.ckpt"))
    checkpoint_path_list = sorted(
        checkpoint_path_list, key=lambda x: x.parent.name, reverse=False
    )
    if len(checkpoint_path_list) == 0:
        return None
    return checkpoint_path_list[-1]

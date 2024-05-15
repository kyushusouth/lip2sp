from pathlib import Path


def get_last_checkpoint_path(checkpoint_dir: Path) -> Path:
    checkpoint_path_list = list(checkpoint_dir.glob("**/*.ckpt"))
    checkpoint_path_list = sorted(
        checkpoint_path_list, key=lambda x: x.parent.name, reverse=False
    )
    return checkpoint_path_list[-1]

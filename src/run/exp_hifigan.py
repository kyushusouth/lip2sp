import subprocess
from pathlib import Path

from src.run.utils import get_last_checkpoint_path


def run_hifigan(
    hifigan_script_path: Path,
    hifigan_checkpoint_dir: Path,
    hifigan_checkpoint_path: Path | None,
    hifigan_input: str,
    group_name: str,
    debug: bool,
):
    if hifigan_checkpoint_path is None:
        subprocess.run(
            [
                "python",
                str(hifigan_script_path),
                "data_choice.kablab.use=false",
                "data_choice.jvs.use=false",
                "data_choice.hifi_captain.use=true",
                "data_choice.jsut.use=false",
                f"model.hifigan.input={hifigan_input}",
                "model.hifigan.freeze=false",
                "training=hifigan_debug" if debug else "training=hifigan",
                f"training.wandb.group_name={group_name}",
                "training.finetune=false",
            ]
        )
    subprocess.run(
        [
            "python",
            str(hifigan_script_path),
            "data_choice.kablab.use=false",
            "data_choice.jvs.use=true",
            "data_choice.hifi_captain.use=false",
            "data_choice.jsut.use=false",
            f"model.hifigan.input={hifigan_input}",
            "model.hifigan.freeze=false",
            "training=hifigan_debug" if debug else "training=hifigan",
            f"training.wandb.group_name={group_name}",
            "training.finetune=true",
            f"training.finetune_start_model_path={str(get_last_checkpoint_path(hifigan_checkpoint_dir)) if hifigan_checkpoint_path is None else hifigan_checkpoint_path}",
        ]
    )


def main():
    debug = True
    hifigan_script_path = Path("/home/minami/lip2sp/src/main/hifigan.py")
    hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/hifigan")
    base_hubert_script_path = Path("/home/minami/lip2sp/src/main/base_hubert.py")
    if debug:
        hifigan_checkpoint_dir = Path("/home/minami/lip2sp/checkpoints/debug_hifigan")

    run_hifigan(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        hifigan_checkpoint_path=None,
        hifigan_input="feature",
        group_name="hifigan",
        debug=debug,
    )
    run_hifigan(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        hifigan_checkpoint_path=None,
        hifigan_input="cat_mel_hubert_encoder",
        group_name="hifigan",
        debug=debug,
    )
    run_hifigan(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        hifigan_checkpoint_path=None,
        hifigan_input="cat_mel_hubert_cluster",
        group_name="hifigan",
        debug=debug,
    )
    run_hifigan(
        hifigan_script_path=hifigan_script_path,
        hifigan_checkpoint_dir=hifigan_checkpoint_dir,
        hifigan_checkpoint_path=None,
        hifigan_input="cat_mel_hubert_encoder_hubert_cluster",
        group_name="hifigan",
        debug=debug,
    )


if __name__ == "__main__":
    main()

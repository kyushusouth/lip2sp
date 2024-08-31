from pathlib import Path
from tqdm import tqdm
import torch

from src.pl_module.hifigan import LitHiFiGANModel
import librosa
import hydra
import omegaconf
import polars as pl


def generator_loss(disc_outputs: torch.Tensor) -> tuple:
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: omegaconf.DictConfig) -> None:
    hifigan_model_path = {
        "feature": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240618_133442/epoch:19-step:26000.ckpt"
        ),
        "feature_hubert_encoder": Path(""),
        "feature_hubert_cluster": Path(""),
        "cat_mel_hubert_encoder": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240619_092235/epoch:26-step:35100.ckpt"
        ),
        "cat_mel_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240620_050910/epoch:23-step:31200.ckpt"
        ),
        "cat_hubert_encoder_hubert_cluster": Path(""),
        "cat_mel_hubert_encoder_hubert_cluster": Path(
            "/home/minami/lip2sp/checkpoints/hifigan/20240621_005533/epoch:19-step:26000.ckpt"
        ),
    }
    metadata_lst = [
        {
            "date": "20240621_134621",
            "hifigan_input": "feature",
        },
        {
            "date": "20240621_155144",
            "hifigan_input": "cat_mel_hubert_encoder",
        },
        {
            "date": "20240621_202419",
            "hifigan_input": "cat_mel_hubert_cluster",
        },
        {
            "date": "20240622_003027",
            "hifigan_input": "cat_mel_hubert_encoder_hubert_cluster",
        },
        {
            "date": "20240622_103111",
            "hifigan_input": "cat_mel_hubert_encoder",
        },
        {
            "date": "20240623_001016",
            "hifigan_input": "cat_mel_hubert_cluster",
        },
        {
            "date": "20240622_161416",
            "hifigan_input": "cat_mel_hubert_encoder_hubert_cluster",
        },
    ]

    results = []

    for metadata in metadata_lst:
        cfg.model.hifigan.input = metadata["hifigan_input"]
        hifigan = LitHiFiGANModel.load_from_checkpoint(
            hifigan_model_path[metadata["hifigan_input"]],
            cfg=cfg,
        )
        hifigan.cuda()
        hifigan.eval()

        data_dir = Path("/home/minami/lip2sp/results/base_hubert") / metadata["date"]
        gt_path_lst = list(data_dir.glob("**/gt.wav"))

        for gt_path in tqdm(gt_path_lst):
            abs_path = str(gt_path).replace(gt_path.stem, "abs")
            pred_path = str(gt_path).replace(gt_path.stem, "pred")
            speaker = gt_path.parents[1].name
            sample = gt_path.parents[0].name

            wav_gt, _ = librosa.load(str(gt_path), sr=cfg.data.audio.sr)
            wav_abs, _ = librosa.load(str(abs_path), sr=cfg.data.audio.sr)
            wav_pred, _ = librosa.load(str(pred_path), sr=cfg.data.audio.sr)

            wav_gt = torch.from_numpy(wav_gt).unsqueeze(0).unsqueeze(0).cuda()
            wav_abs = torch.from_numpy(wav_abs).unsqueeze(0).unsqueeze(0).cuda()
            wav_pred = torch.from_numpy(wav_pred).unsqueeze(0).unsqueeze(0).cuda()

            with torch.no_grad():
                y_df_hat_r_abs, y_df_hat_g_abs, _, _ = hifigan.mpd(wav_gt, wav_abs)
                y_ds_hat_r_abs, y_ds_hat_g_abs, _, _ = hifigan.msd(wav_gt, wav_abs)
                y_df_hat_r_pred, y_df_hat_g_pred, _, _ = hifigan.mpd(wav_gt, wav_pred)
                y_ds_hat_r_pred, y_ds_hat_g_pred, _, _ = hifigan.msd(wav_gt, wav_pred)

            loss_gen_f_abs, losses_gen_f_abs = generator_loss(y_df_hat_g_abs)
            loss_gen_s_abs, losses_gen_s_abs = generator_loss(y_ds_hat_g_abs)
            loss_gen_f_pred, losses_gen_f_pred = generator_loss(y_df_hat_g_pred)
            loss_gen_s_pred, losses_gen_s_pred = generator_loss(y_ds_hat_g_pred)

            results.append(
                [
                    metadata["date"],
                    metadata["hifigan_input"],
                    speaker,
                    sample,
                    "abs",
                    loss_gen_f_abs.item(),
                    loss_gen_s_abs.item(),
                ]
            )
            results.append(
                [
                    metadata["date"],
                    metadata["hifigan_input"],
                    speaker,
                    sample,
                    "pred",
                    loss_gen_f_pred.item(),
                    loss_gen_s_pred.item(),
                ]
            )

    df = pl.DataFrame(
        data=results,
        schema=[
            "date",
            "hifigan_input",
            "speaker",
            "sample",
            "kind",
            "loss_gen_f",
            "loss_gen_s",
        ],
    )


if __name__ == "__main__":
    main()

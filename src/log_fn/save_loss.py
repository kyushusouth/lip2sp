import numpy as np
import wandb


def save_epoch_loss_plot(
    title: str, train_loss_list: list, val_loss_list: list
) -> None:
    wandb.log(
        {
            title: wandb.plot.line_series(
                xs=np.arange(len(train_loss_list)),
                ys=[train_loss_list, val_loss_list],
                keys=["train loss", "validation loss"],
                title=title,
                xname="epoch",
            )
        }
    )

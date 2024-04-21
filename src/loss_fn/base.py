import torch
import torch.nn.functional as F


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    masking the padding parts
    """
    device = lengths.device
    if not isinstance(lengths, list):
        lengths = lengths.tolist()

    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask.unsqueeze(1).to(device=device)  # (B, 1, T)


class LossFunctions:
    def __init__(self, weight=None, use_weighted_mean=False):
        self.weight = weight
        self.use_weighted_mean = use_weighted_mean

    def mse_loss(self, output, target, data_len, max_len):
        """
        output, target : (B, C, T) or (B, C, H, W, T)
        """
        mask = make_pad_mask(data_len, max_len)
        loss = (output - target) ** 2
        if loss.dim() == 5:
            loss = torch.mean(loss, dim=(2, 3))

        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=1)  # (B, T)
        mask = mask.squeeze(1)  # (B, T)
        n_loss = torch.where(
            mask == 0,
            torch.ones_like(mask).to(torch.float32),
            torch.zeros_like(mask).to(torch.float32),
        )
        mse_loss = torch.sum(loss) / torch.sum(n_loss)
        return mse_loss

    def l1_loss(self, pred, target, mask):
        """
        pred: (B, C, T)
        target: (B, C, T)
        mask (True elements are ignored.): (B, T)
        """
        loss_list = []
        for i in range(mask.shape[0]):
            pred_unmasked = pred[i, :, ~mask[i, :]]
            target_unmasked = target[i, :, ~mask[i, :]]
            loss = torch.nn.functional.l1_loss(
                pred_unmasked, target_unmasked, reduction="none"
            )
            loss_list.append(loss)
        loss = torch.cat(loss_list, dim=-1).mean()
        return loss

    def mae_loss(self, output, target, data_len, max_len):
        mask = make_pad_mask(data_len, max_len)
        loss = torch.abs((output - target))
        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=1)  # (B, T)
        mask = mask.squeeze(1)  # (B, T)
        n_loss = torch.where(
            mask == 0,
            torch.ones_like(mask).to(torch.float32),
            torch.zeros_like(mask).to(torch.float32),
        )
        loss = torch.sum(loss) / torch.sum(n_loss)
        return loss

    def bce_with_logits_loss(self, output, target, data_len, max_len):
        mask = make_pad_mask(data_len, max_len)
        loss = F.binary_cross_entropy_with_logits(output, target, reduction="none")
        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        n_loss = torch.where(
            mask == 0,
            torch.ones_like(mask).to(torch.float32),
            torch.zeros_like(mask).to(torch.float32),
        )
        loss = torch.sum(loss) / torch.sum(n_loss)
        return loss

import torch.nn.functional as F


def single_part_loss(model_output, gt, model=None, **kwargs):
    """
    Calculates occupancy loss for a single part.
    """
    pred_sdf = model_output["model_out"]
    gt_sdf = gt["sdf"]

    # Normalize gt_sdf from [-1, 1] to [0, 1]
    gt_sdf = (gt_sdf + 1) / 2

    # Use BCE with logits
    loss = F.binary_cross_entropy_with_logits(
        pred_sdf.squeeze(-1), gt_sdf.squeeze(-1), reduction="none"
    )
    return {"occupancy": loss.mean()}

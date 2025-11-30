import torch.nn.functional as F

def single_part_loss(model_output, gt, model=None, **kwargs):
    """
    Calculates occupancy loss for a single part using BCE with logits.
    The ground truth 'sdf' from the dataloader is assumed to be binary (0 or 1).
    """
    # This might be just occ_sigmoid without moving
    pred_sdf = model_output["model_out"]
    gt_sdf = gt["sdf"]

    loss = F.binary_cross_entropy_with_logits(
        pred_sdf.squeeze(-1), gt_sdf.squeeze(-1), reduction="none"
    )

    return {"occupancy": loss.sum(-1).mean()}

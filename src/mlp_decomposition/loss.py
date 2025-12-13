import torch.nn.functional as F

def single_part_loss(model_output, gt, model=None, **kwargs):
    """
    Calculates occupancy loss for a single part using BCE with logits.
    The ground truth 'sdf' from the dataloader is assumed to be binary (0 or 1).
    """
    #TODO Implement Vectorized loss
    # [0, 1, 2] -> Part-MLP -> 0/1
    # MLP -> thresholden > 0.5 -> E(MLP1 or MLP2 or MLP3 or MLP4) 

    # This might be just occ_sigmoid without moving
    pred_sdf = model_output["model_out"]
    gt_sdf = gt["sdf"]

    loss = F.binary_cross_entropy_with_logits(
        pred_sdf.squeeze(-1), gt_sdf.squeeze(-1), reduction="none"
    )

    return {"occupancy": loss.sum(-1).mean()}

#TODO: Check this and then make it modular
plane_labels = {"body": 1, "wing": 2, "tail": 3, "engine": 4}
def all_part_loss(model_output, gt, model=None, **kwargs):
    """
    Calculates occupancy loss for a all parts using BCE with logits.
    The ground truth 'semantic_label' from the dataloader is assumed to be the 
    part label [0,n_parts], where 0 is unoccupied space and 1-n is the part
    """

    # Global Occ, currently not needed
    #pred_sdf = model_output["model_out"]
    #Label
    gt_label = gt["semantic_label"]
    
    total_loss = 0
    part_losses = {}
    for part, pred_occ in model_output["parts"].items():
        part_occ = (gt_label == plane_labels[part]).float()
        # This is equivalent to loss before
        part_loss = F.binary_cross_entropy_with_logits(
            pred_occ.squeeze(-1), part_occ, reduction="none"
        ).sum(-1).mean()
        part_losses[part] = part_loss

        total_loss += part_loss

    return {"occupancy": total_loss, "part_losses": part_losses}

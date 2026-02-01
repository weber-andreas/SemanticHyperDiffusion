"""Implements a generic training loop."""

import os
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm.autonotebook import tqdm

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def train(
    model,
    train_dataloader,
    epochs,
    lr,
    steps_til_summary,
    model_dir,
    loss_fn,
    summary_fn,
    wandb,
    val_dataloader=None,
    double_precision=False,
    clip_grad=False,
    use_lbfgs=False,
    loss_schedules=None,
    filename=None,
    cfg=None,
):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    if cfg.scheduler.type == "step":
        scheduler = StepLR(
            optim,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma,
            verbose=True,
        )
    elif cfg.scheduler.type == "adaptive":
        scheduler = ReduceLROnPlateau(
            optim,
            patience=cfg.scheduler.patience_adaptive,
            factor=cfg.scheduler.factor,
            threshold=cfg.scheduler.threshold,
            min_lr=cfg.scheduler.min_lr,
        )

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(
            lr=lr,
            params=model.parameters(),
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

    os.makedirs(model_dir, exist_ok=True)
    checkpoints_dir = model_dir

    total_steps = 0
    best_loss = float("inf")
    patience = cfg.scheduler.patience
    num_bad_epochs = 0
    labels = {part_name: idx + 1 for idx, part_name in enumerate(cfg.label_names)}
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):

            total_loss, total_items = 0, 0
            part_loss_accumulators = {}

            for step, (model_input, gt) in enumerate(train_dataloader):
                # BUG: This enumerate does go on indefinetly without the following if
                # TODO: Figure out why this is needed
                if step >= len(train_dataloader):
                    break
                start_time = time.time()

                model_input = {
                    key: value.to(DEVICE) for key, value in model_input.items()
                }
                gt = {key: value.to(DEVICE) for key, value in gt.items()}

                if double_precision:
                    model_input = {
                        key: value.double() for key, value in model_input.items()
                    }
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:

                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt, labels)
                        train_loss = 0.0
                        for loss_name, loss in losses.items():
                            if loss_name != "part_losses":
                                train_loss += loss.mean()
                        train_loss.backward()
                        return train_loss

                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt, labels, model)

                train_loss = 0.0
                for loss_name, loss in losses.items():
                    if loss_name != "part_losses":
                        # This should be in loss function?
                        single_loss = loss.mean()
                        if loss_schedules is not None and loss_name in loss_schedules:
                            single_loss *= loss_schedules[loss_name](total_steps)

                        train_loss += single_loss

                train_losses.append(train_loss.item())
                # Changed to be more universal
                batch_size = len(model_input["coords"])
                total_loss += train_loss.item() * batch_size
                total_items += batch_size

                if "part_losses" in losses and isinstance(losses["part_losses"], dict):
                    for p_name, p_loss in losses["part_losses"].items():
                        if p_name not in part_loss_accumulators:
                            part_loss_accumulators[p_name] = 0.0

                        # Accumulate: loss_val * batch_size (matching total_loss logic)
                        part_loss_accumulators[p_name] += (
                            p_loss.mean().item() * batch_size
                        )

                if not total_steps % steps_til_summary:
                    pass

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=clip_grad
                            )

                    optim.step()

                pbar.update(1)
                pbar.set_description(
                    "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                    % (epoch, train_loss.item(), time.time() - start_time)
                )
                total_steps += 1

            epoch_loss = total_loss / total_items
            if cfg.scheduler.type == "adaptive":
                prev_bad_epochs = scheduler.num_bad_epochs
                prev_best = scheduler.best
                prev_lr = next(iter(optim.param_groups))["lr"]
                scheduler.step(epoch_loss)
                curr_bad_epochs = scheduler.num_bad_epochs
                new_lr = next(iter(optim.param_groups))["lr"]
                new_best = scheduler.best
            else:
                scheduler.step()
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
            optim.param_groups[0]["lr"] = max(
                optim.param_groups[0]["lr"], cfg.scheduler.min_lr
            )
            if cfg.strategy == "continue":
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoints_dir, f"{filename}_model_final_{epoch}.pth"
                    ),
                )

            log_data = {
                "epoch_loss": epoch_loss,
                "lr": optim.param_groups[0]["lr"],
                "epoch": epoch,
            }

            for p_name, p_total_val in part_loss_accumulators.items():
                log_data[f"part_loss_{p_name}"] = p_total_val / total_items

            wandb.log(log_data)

            if num_bad_epochs == patience:
                break
        if not cfg.mlp_config.move:
            summary_fn(
                "audio_samples",
                model,
                model_input,
                gt,
                model_output,
                wandb,
                total_steps,
            )
        final_log_data = {"total_train_loss": train_loss.item()}
        if "part_losses" in losses and isinstance(losses["part_losses"], dict):
            for p_name, p_loss in losses["part_losses"].items():
                final_log_data[f"total_part_loss_{p_name}"] = p_loss.mean().item()

        wandb.log(final_log_data)

        if cfg.strategy != "continue":
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, f"{filename}_model_final.pth"),
            )
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #            np.array(train_losses))


class LinearDecaySchedule:
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(
            iter / self.num_steps, 1.0
        )

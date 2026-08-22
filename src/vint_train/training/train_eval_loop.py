import wandb
import os
import tempfile
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad
from vint_train.training.train_utils import train_dino, evaluate_dino
from vint_train.training.train_utils import train_dvn, evaluate_dvn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.training_utils import EMAModel


def atomic_torch_save(obj, path: str) -> None:
    """Write a checkpoint completely before replacing the destination."""
    directory = os.path.dirname(path) or "."
    basename = os.path.basename(path)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{basename}.", suffix=".tmp", dir=directory
    )
    os.close(fd)
    try:
        torch.save(obj, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    aconfig: Dict[str, object] = {},
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                # aconfig=aconfig,
            )

        avg_test_dist_loss, avg_test_action_loss, avg_total_test_loss = [], [], []

        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                # aconfig=aconfig,
            )

            avg_test_dist_loss.append(test_dist_loss)
            avg_test_action_loss.append(test_action_loss)
            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        # log average eval loss
        # wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
        # wandb.log({
        #     "avg_total_test_loss": np.mean(avg_total_test_loss),
        #     "lr": optimizer.param_groups[0]["lr"],
        # }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

    # Flush the last set of eval logs
    # wandb.log({})
    print('test_dist_loss', avg_test_dist_loss)
    print('test_action_loss', avg_test_action_loss)
    print('total_test_loss', avg_total_test_loss)
    print(f'avg_test_dist_loss: {np.mean(avg_test_dist_loss)}, avg_test_action_loss: {np.mean(avg_test_action_loss)}, avg_total_test_loss: {np.mean(avg_total_test_loss)}')


def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: None, # DDPMScheduler
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    # ema_model = EMAModel(model=model,power=0.75)
    ema_model = None
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        atomic_torch_save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        atomic_torch_save(model.state_dict(), numbered_path)
        atomic_torch_save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        atomic_torch_save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        atomic_torch_save(lr_scheduler.state_dict(), latest_scheduler_path)


        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # Flush the last set of eval logs
    wandb.log({})
    print()

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params

class EMAMeter:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None

    def update(self, x):
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * x

    def get(self, default=1.0):
        return self.value if self.value is not None else default

def train_eval_loop_dino(
    train_model: bool,
    model: nn.Module,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    learn_angle: bool = True,
    alpha: float = 0.5,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    ema_meters = {
        "dist": EMAMeter(0.99),
        "mhp":  EMAMeter(0.99),
        "lfp":  EMAMeter(0.99),
        "col":  EMAMeter(0.99),
        "floor": 1e-3,
        "eps": 1e-8
    }
        
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start DiNOv2 Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_dino(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                ema_meters=ema_meters,
            )
        
        avg_test_dist_loss, avg_test_action_loss, avg_total_test_loss = [], [], []

        for dataset_type, dataset_loader in test_dataloaders.items():
            print(
                f"Start {dataset_type} DiNOv2 Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            test_dist_loss, test_action_loss, total_eval_loss = evaluate_dino(
                eval_type=dataset_type,
                model=model,
                dataloader=dataset_loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                ema_meters=ema_meters,
            )
            avg_test_dist_loss.append(test_dist_loss)
            avg_test_action_loss.append(test_action_loss)
            avg_total_test_loss.append(total_eval_loss)
        
        # Save Checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()

        # Save latest
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        atomic_torch_save(checkpoint, latest_path)
        atomic_torch_save(checkpoint, numbered_path)
        
    print('test_dist_loss', avg_test_dist_loss)
    print('test_action_loss', avg_test_action_loss)
    print('total_test_loss', avg_total_test_loss)
    print(f'avg_test_dist_loss: {np.mean(avg_test_dist_loss)}, avg_test_action_loss: {np.mean(avg_test_action_loss)}, avg_total_test_loss: {np.mean(avg_total_test_loss)}')


def train_eval_loop_dvn(
    train_model: bool,
    model: nn.Module,
    optimizer: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    learn_angle: bool = True,
    alpha: float = 0.5,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    sigreg_weight: float = 0.0,
    debug_intermediates: bool = False,
):

    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"dvn_latest.pth")

    ema_meters = {
        "dist": EMAMeter(0.99),
        "mhp":  EMAMeter(0.99),
        # "mhp_pos":  EMAMeter(0.99),
        # "mhp_angle":  EMAMeter(0.99),
        # "mhp_score":  EMAMeter(0.99),
        "lfp":  EMAMeter(0.99),
        "floor": 1e-3
    }
        
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start DVN Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_dvn(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                ema_meters=ema_meters,
                sigreg_weight=sigreg_weight,
                debug_intermediates=debug_intermediates,
            )
        
        avg_test_dist_loss, avg_test_mhp_loss, avg_test_lfp_loss, avg_total_test_loss = [], [], [], []

        for dataset_type, dataset_loader in test_dataloaders.items():
            print(
                f"Start {dataset_type} DVN Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            # test_dist_loss, test_pos_loss, test_angle_loss, total_eval_loss
            test_dist_loss, test_mhp_loss, test_lfp_loss, total_eval_loss = evaluate_dvn(
                eval_type=dataset_type,
                model=model,
                dataloader=dataset_loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                ema_meters=ema_meters,
                sigreg_weight=sigreg_weight,
                debug_intermediates=debug_intermediates,
            )
            avg_test_dist_loss.append(test_dist_loss)
            avg_test_mhp_loss.append(test_mhp_loss)
            avg_test_lfp_loss.append(test_lfp_loss)
            avg_total_test_loss.append(total_eval_loss)
        
        # Save Checkpoints
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()

        # Save latest
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path) 
        
    print('test_dist_loss', avg_test_dist_loss)
    print('avg_test_mhp_loss', avg_test_mhp_loss)
    print('avg_test_lfp_loss', avg_test_lfp_loss)
    print('total_test_loss', avg_total_test_loss)
    print(f'avg_test_dist_loss: {np.mean(avg_test_dist_loss)}, avg_test_mhp_loss: {np.mean(avg_test_mhp_loss)}, avg_test_lfp_loss: {np.mean(avg_test_lfp_loss)}, avg_total_test_loss: {np.mean(avg_total_test_loss)}')

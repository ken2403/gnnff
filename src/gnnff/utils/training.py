import os

import schnetpack as spk
import torch
from torch.optim import Adam

from gnnff.train.trainer import Trainer

__all__ = ["get_metrics", "get_trainer", "simple_loss_fn"]


def get_metrics(args):
    property_keys = args.args.predict_property.keys()
    metrics = []
    if "forces" in property_keys:
        metrics.append(
            spk.train.metrics.MeanAbsoluteError(
                args.predict_property["forces"],
                args.predict_property["forces"],
                element_wise=True,
            ),
            spk.train.metrics.RootMeanSquaredError(
                args.predict_property["forces"],
                args.predict_property["forces"],
                element_wise=True,
            ),
        )
    elif "energy" in property_keys:
        metrics.append(
            spk.train.metrics.MeanAbsoluteError(
                args.predict_property["energy"],
                args.predict_property["energy"],
            ),
            spk.train.metrics.RootMeanSquaredError(
                args.predict_property["energy"],
                args.predict_property["energy"],
            ),
        )
    return metrics


def get_trainer(args, model, train_loader, val_loader, metrics):
    # setup optimizer
    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    # setup hook and logging
    hooks = [spk.train.MaxEpochHook(args.max_epochs)]
    if args.max_steps:
        hooks.append(spk.train.MaxStepHook(max_steps=args.max_steps))

    schedule = spk.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
    hooks.append(schedule)

    if args.logger == "csv":
        logger = spk.train.CSVHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)
    elif args.logger == "tensorboard":
        logger = spk.train.TensorboardHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    # setup loss function
    loss_fn = simple_loss_fn(args)

    # setup trainer
    trainer = Trainer(
        args.modelpath,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        hooks=hooks,
        regularization=args.regularization,
    )
    return trainer


def simple_loss_fn(args):
    def loss(batch, result):
        property_name = args.predict_property[list(args.predict_property.keys())[0]]
        diff = batch[property_name] - result[property_name]
        diff = diff ** 2
        err_sq = torch.mean(diff)
        return err_sq

    return loss

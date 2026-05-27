"""Epoch-based training APIs for :class:`BasicTrainer`."""

from __future__ import annotations

import warmup_scheduler


def train_validation_epochs(trainer, lr, n, warmup_epochs=0):
    optimizer, scheduler, scaler = trainer._get_optimizer_and_scheduler(lr, n)

    if trainer.recipe is not None:
        warmup_epochs = 0

    if warmup_epochs > 0:
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer,
            multiplier=1.0,
            total_epoch=warmup_epochs,
            after_scheduler=scheduler,
        )

    validation_accuracy = 0.0
    for _ in range(int(n) + int(warmup_epochs)):
        training_accuracy = trainer._train_one_epoch(optimizer, scheduler, scaler)
        trainer._report("Training accuracy", training_accuracy)
        validation_accuracy = trainer.validate()

    return validation_accuracy


def train_until_target_accuracy(trainer, lr, max_epochs, target_accuracy, warmup_epochs):
    optimizer, scheduler, scaler = trainer._get_optimizer_and_scheduler(lr, max_epochs)

    if trainer.recipe is not None:
        warmup_epochs = 0

    if warmup_epochs > 0:
        scheduler = warmup_scheduler.GradualWarmupScheduler(
            optimizer, multiplier=1., total_epoch=warmup_epochs, after_scheduler=scheduler
        )

    validation_accuracy = 0.0
    for _ in range(max_epochs + warmup_epochs):
        training_accuracy = trainer._train_one_epoch(optimizer, scheduler, scaler)
        trainer._report("Training accuracy", training_accuracy)

        validation_accuracy = trainer.validate()
        if validation_accuracy >= target_accuracy:
            trainer._train_one_epoch(optimizer, scheduler, scaler)
            trainer._train_one_epoch(optimizer, scheduler, scaler)
            validation_accuracy = trainer.validate()
            break

    trainer.test()
    return validation_accuracy

"""Training configuration for Sup3r models"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for GAN training.

    This dataclass consolidates all training parameters to simplify the
    train() method signature and make it easier to manage training configs.

    Parameters
    ----------
    n_epoch : int
        Number of epochs to train on
    weight_gen_advers : float
        Weight factor for the adversarial loss component of the generator
        vs. the discriminator.
    train_gen : bool
        Flag whether to train the generator for this set of epochs
    train_disc : bool
        Flag whether to train the discriminator for this set of epochs
    disc_loss_bounds : tuple
        Lower and upper bounds for the discriminator loss outside of which
        the discriminator will not train unless train_disc=True and
        train_gen=False.
    checkpoint_int : int | None
        Epoch interval at which to save checkpoint models.
    out_dir : str
        Directory to save checkpoint GAN models. Should have {epoch} in
        the directory name. This directory will be created if it does not
        already exist.
    early_stop_on : str | None
        If not None, this should be a column in the training history to
        evaluate for early stopping (e.g. validation_loss_gen,
        validation_loss_disc).
    early_stop_threshold : float
        The absolute relative fractional difference in validation loss
        between subsequent epochs below which an early termination is
        warranted.
    early_stop_n_epoch : int
        The number of consecutive epochs that satisfy the threshold that
        warrants an early stop.
    adaptive_update_bounds : tuple
        Tuple specifying allowed range for loss_details[comparison_key].
    adaptive_update_fraction : float
        Amount by which to increase or decrease adversarial weights for
        adaptive updates
    multi_gpu : bool
        Flag to break up the batch for parallel gradient descent
        calculations on multiple gpus.
    log_tb : bool
        Whether to write log file for use with tensorboard.
    export_tb : bool
        Whether to export profiling information to tensorboard.
    swa_start : int | None
        Epoch to start SWA (e.g., int(0.75 * n_epoch)). If None, SWA is
        disabled.
    swa_freq : int
        How often to update SWA (1 = every epoch)
    swa_lr : float | None
        Constant learning rate for SWA phase (if None, keeps current schedule)
    swa_bn_update_batches : int
        Number of batches to use for updating batch normalization statistics
        after swapping to SWA weights. (Only used if model has batch
        normalization layers and swa_start is not None)
    """

    n_epoch: int
    weight_gen_advers: float = 0.001
    train_gen: bool = True
    train_disc: bool = True
    disc_loss_bounds: tuple[float, float] = (0.45, 0.6)
    checkpoint_int: Optional[int] = None
    out_dir: str = './gan_{epoch}'
    early_stop_on: Optional[str] = None
    early_stop_threshold: float = 0.005
    early_stop_n_epoch: int = 5
    adaptive_update_bounds: tuple[float, float] = (0.9, 0.99)
    adaptive_update_fraction: float = 0.0
    multi_gpu: bool = False
    log_tb: bool = False
    export_tb: bool = False
    swa_start: Optional[int] = None
    swa_freq: int = 1
    swa_lr: Optional[float] = None
    swa_bn_update_batches: int = 100

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_epoch <= 0:
            raise ValueError(f'n_epoch must be positive, got {self.n_epoch}')

        if self.swa_start is not None:
            if self.swa_start < 0 or self.swa_start >= self.n_epoch:
                raise ValueError(
                    f'swa_start must be between 0 and n_epoch '
                    f'({self.n_epoch}), got {self.swa_start}'
                )
            if self.swa_freq <= 0:
                raise ValueError(
                    f'swa_freq must be positive, got {self.swa_freq}'
                )

        if '{epoch}' not in self.out_dir and self.checkpoint_int is not None:
            raise ValueError(
                f"out_dir must contain '{{epoch}}' when checkpoint_int is "
                f'set, got {self.out_dir}'
            )

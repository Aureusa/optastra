import logging

from torch import nn

from .base import Hook


class FreezeBackboneHook(Hook):
    def __init__(self, freeze: bool = True, eval_mode: bool = True):
        """
        Hook to freeze or unfreeze the backbone of the model during training.
        This hook should be added to the trainer before training starts.

        :param freeze: If True, freeze the backbone; if False, unfreeze it.
        :param eval_mode: With ``freeze=True``, also keep the backbone in eval mode
            for the whole run (the standard linear-probe protocol of SimCLR, MoCo,
            BYOL): BatchNorm then normalises with, and no longer updates, its
            pretrained running statistics, and dropout / stochastic depth inside the
            backbone are off. ``requires_grad=False`` alone does not stop BatchNorm's
            running statistics from adapting to the new data in train mode. The
            Trainer puts the model back in train mode after every evaluation, so
            eval mode is re-applied before every step. Pass ``eval_mode=False`` for
            the old behaviour (frozen weights, BatchNorm statistics still updating).
        """
        self.freeze = freeze
        self.eval_mode = eval_mode
        self._backbones: list[nn.Module] = []
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)

    def before_train(self, state):
        """
        Freeze or unfreeze the backbone of the model based on the `freeze` parameter.
        """
        from optastra.backbones import Backbone  # Import here to avoid circular import issues
        # Checks if a model has an atribute that is a Backbone class instance, and if so, freezes or unfreezes it.
        self._backbones = []
        for name, module in state.model.named_modules():
            if isinstance(module, Backbone):
                for param in module.parameters():
                    param.requires_grad = not self.freeze
                self._backbones.append(module)
                mode = " (eval mode: BatchNorm statistics fixed)" if self.freeze and self.eval_mode else ""
                self.logger.info(f"{'Frozen' if self.freeze else 'Unfrozen'} backbone: {name}{mode}")
        self._apply_eval_mode()

    def before_step(self, state):
        # model.train() (e.g. after Trainer.evaluate) also switches the backbone back to train mode.
        self._apply_eval_mode()

    def _apply_eval_mode(self) -> None:
        if self.freeze and self.eval_mode:
            for backbone in self._backbones:
                if backbone.training:
                    backbone.eval()

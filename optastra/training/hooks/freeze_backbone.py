import logging
from .base import Hook


class FreezeBackboneHook(Hook):
    def __init__(self, freeze: bool = True):
        """
        Hook to freeze or unfreeze the backbone of the model during training.
        This hook should be added to the trainer before training starts.

        :param freeze: If True, freeze the backbone; if False, unfreeze it.
        """
        self.freeze = freeze
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)

    def before_train(self, state):
        """
        Freeze or unfreeze the backbone of the model based on the `freeze` parameter.
        """
        from optastra.backbones import Backbone  # Import here to avoid circular import issues
        # Checks if a model has an atribute that is a Backbone class instance, and if so, freezes or unfreezes it.
        for name, module in state.model.named_modules():
            if isinstance(module, Backbone):
                for param in module.parameters():
                    param.requires_grad = not self.freeze
                self.logger.info(f"{'Frozen' if self.freeze else 'Unfrozen'} backbone: {name}")

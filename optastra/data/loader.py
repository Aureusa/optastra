from torch.utils.data import DataLoader
from .collate import CollateFn

from ..tasks import Task


def build_dataloader(dataset, task: "Task", batch_size: int, collate_kwargs: dict = None, **kwargs) -> DataLoader:
    """
    The dataloader wires itself to whatever collate the task declares --
    the user never has to remember 'detection needs collate_fn=my_ragged_collate'.
    """
    return DataLoader(dataset, batch_size=batch_size,
                       collate_fn=CollateFn.create(task.collate, **(collate_kwargs or {})), **kwargs)

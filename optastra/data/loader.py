from torch.utils.data import DataLoader
from ._registry import get_collate_entrypoint

from ..tasks import Task


def build_dataloader(dataset, task: "Task", batch_size: int, **kwargs) -> DataLoader:
    """
    The dataloader wires itself to whatever collate the task declares --
    the user never has to remember 'detection needs collate_fn=my_ragged_collate'.
    """
    return DataLoader(dataset, batch_size=batch_size,
                       collate_fn=get_collate_entrypoint(task.collate), **kwargs)

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader
from .collate import CollateFn

if TYPE_CHECKING:  # only for the annotation -- importing tasks here would create an import cycle
    from ..tasks import Task


def build_dataloader(dataset, task: "Task", batch_size: int, collate_kwargs: dict = None, **kwargs) -> DataLoader:
    """
    The dataloader wires itself to whatever collate the task declares --
    the user never has to remember 'detection needs collate_fn=my_ragged_collate'.

    ``collate_kwargs`` are bound to the collate function, e.g.
    ``collate_kwargs={"size_divisibility": 32}`` for the ragged collate.
    """
    return DataLoader(dataset, batch_size=batch_size,
                       collate_fn=CollateFn.create(task.collate, **(collate_kwargs or {})), **kwargs)

from .base import *
from .classification import *

from ._registry import (
    register_task,
    list_tasks,
    get_task_entrypoint,
    get_task_module,
    get_task_default_config,
    check_task_registered,
)

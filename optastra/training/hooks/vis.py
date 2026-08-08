import torch, os
from .base import Hook
from ..state import TrainerState

from ...visualization.visualizer import Visualizer as Vis


class VisualizerHook(Hook):
    def __init__(self, output_dir: str, visualize_every: int = 500):
        self.output_dir = output_dir
        self.visualize_every = visualize_every
        self.vis = None
        os.makedirs(output_dir, exist_ok=True)

    def before_step(self, state: TrainerState) -> None:
        if self.vis is None:
            self.vis = Vis(nrows=1, ncols=2, figsize=(10, 5))  # Adjust nrows and ncols as needed
        state.current_batch = state.current_batch  # Ensure current_batch is set before visualization

        # Take the first image from the batch for visualization
        image = state.current_batch['image'][0].cpu().numpy().transpose(1, 2, 0)  # Assuming image is in (C, H, W) format

        self.vis.draw_image(image, title=f"Pre Transform (it={state.iter})", row=0, col=0)

    def after_step(self, state: TrainerState) -> None:
        if state.iter == 0 or state.iter % self.visualize_every != 0:
            return

        image = state.current_batch['image'][0].cpu().numpy().transpose(1, 2, 0)  # Assuming image is in (C, H, W) format
        self.vis.draw_image(image, title=f"Post Transform (it={state.iter})", row=0, col=1)
        self.vis.save(os.path.join(self.output_dir, f"visualization_iter_{state.iter}.png"))
        
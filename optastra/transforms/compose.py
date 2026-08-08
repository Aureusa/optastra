from .base import Transform


__all__ = ["Compose"]


class Compose(Transform):
    """Not registered -- built directly, since its 'config' is a list of
    other Transforms, not a flat dataclass. Same reasoning as why
    nn.Sequential isn't a registered component."""
    def __init__(self, transforms: list[Transform]):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    
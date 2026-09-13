"""Discoverability: answer "what backbones/transforms/tasks exist?" from
the shell without reading source.

Run with:
    python examples/04_list_and_describe_components.py
"""
from optastra import Backbone, Transform, list_all_registered_components, list_all_registered_families


def main() -> None:
    print("registered component families:")
    for family in list_all_registered_families():
        print(f"  - {family}")

    print("\nall registered backbones:")
    for name in Backbone.list_all():
        print(f"  - {name}")

    print("\ntransforms with 'crop' in the name:")
    for name in Transform.list_all(filter="crop"):
        print(f"  - {name}")

    print("\nevery family + component (list_all_registered_components):")
    for family, names in list_all_registered_components().items():
        print(f"  {family}: {names}")

    print("\ndefault config for 'resnet50':")
    Backbone.describe("resnet50")


if __name__ == "__main__":
    main()

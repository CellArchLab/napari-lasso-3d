__version__ = "0.0.1"

from ._reader import napari_get_reader
from ._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)

__all__ = (
    "napari_get_reader",
    "ExampleQWidget",
    "ImageThreshold",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
)

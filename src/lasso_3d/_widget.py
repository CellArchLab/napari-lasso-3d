from typing import List

import dijkstra3d
import napari
import numpy as np
from magicgui import magicgui
from membrain_seg.segmentation.dataloading.data_utils import store_tomogram
from napari.layers.shapes._shapes_constants import Mode
from napari.layers.shapes._shapes_mouse_bindings import add_path_polygon_lasso
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import label

from lasso_3d.lasso_add_slices import mask_via_extension
from lasso_3d.shapes_overwrites import redefine_shapelayer_functions


class Lasso3D(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.annotation_box = QHBoxLayout()
        btn_freehand = QPushButton("Freehand")
        btn_freehand.clicked.connect(self._on_click_freehand)
        btn_points = QPushButton("Points")
        btn_points.clicked.connect(self._on_click_polygon)
        self.annotation_box.addWidget(btn_freehand)
        self.annotation_box.addWidget(btn_points)

        self.selection_box = QHBoxLayout()
        self._layer_selection_widget = magicgui(
            self._lasso_from_polygon,
            points_layer={"choices": self._get_valid_points_layers},
            image_layer={"choices": self._get_valid_image_layers},
            call_button="Lasso",
        )
        self.selection_box.addWidget(self._layer_selection_widget.native)

        self.mask_seg_box = QHBoxLayout()
        self._layer_selection_widget_mask = magicgui(
            self._mask_volume,
            image_layer={"choices": self._get_valid_image_layers},
            mask_layer={"choices": self._get_valid_mask_layers},
            masking={"choices": ["isolate", "subtract"]},
            call_button="Mask Volume",
        )
        self.mask_seg_box.addWidget(self._layer_selection_widget_mask.native)

        self.connected_components_box = QHBoxLayout()
        self._layer_selection_widget_connected_components = magicgui(
            self._connected_components,
            mask_layer={"choices": self._get_valid_image_layers},
            remove_small_objects_size={
                "value": 100,
                "widget_type": "SpinBox",
                "min": 0,
                "max": 100000,
            },
            call_button="Connected Components",
        )
        self.connected_components_box.addWidget(
            self._layer_selection_widget_connected_components.native
        )

        self.display_connected_components_box = QHBoxLayout()
        self._layer_selection_widget_display_connected_components = magicgui(
            self._display_connected_components,
            components_layer={"choices": self._get_valid_image_layers},
            component_number={"value": 1},
            call_button="Display Connected Components",
        )
        self.display_connected_components_box.addWidget(
            self._layer_selection_widget_display_connected_components.native
        )

        self.store_tomogram_box = QHBoxLayout()
        self.store_tomogram_widget = magicgui(
            self._store_tomogram,
            image_layer={"choices": self._get_valid_image_layers},
            store_component_number={"value": 1, "label": "Component Number"},
            filename={
                "widget_type": "FileEdit",
                "mode": "d",
                "label": "Folder Path",
            },
            call_button="Store Tomogram",
        )
        self.store_tomogram_box.addWidget(self.store_tomogram_widget.native)

        self.store_all_components_box = QHBoxLayout()
        self.store_all_components_widget = magicgui(
            self._store_all_components,
            image_layer={"choices": self._get_valid_image_layers},
            foldername={
                "widget_type": "FileEdit",
                "mode": "d",
                "label": "Folder Path",
            },
            call_button="Store All Components",
        )
        self.store_all_components_box.addWidget(
            self.store_all_components_widget.native
        )

        self.color_distances_box = QHBoxLayout()
        color_point = QPushButton("Points")
        color_point.clicked.connect(self._on_click_color_point)

        self.color_distances_widget = magicgui(
            self._color_distances,
            image_layer={"choices": self._get_valid_mask_layers},
            point={"value": [0, 0, 0], "label": "Point"},
            connected_component_number={
                "value": 1,
                "label": "Component Number",
            },
            call_button="Color Distances",
        )
        self.color_distances_box.addWidget(color_point)
        self.color_distances_box.addWidget(self.color_distances_widget.native)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(self.annotation_box)
        self.layout().addLayout(self.selection_box)
        self.layout().addLayout(self.mask_seg_box)
        self.layout().addLayout(self.connected_components_box)
        self.layout().addLayout(self.display_connected_components_box)
        self.layout().addLayout(self.store_tomogram_box)
        self.layout().addLayout(self.store_all_components_box)
        self.layout().addLayout(self.color_distances_box)

        viewer.layers.events.inserted.connect(self._on_layer_change)
        viewer.layers.events.removed.connect(self._on_layer_change)

    def _on_layer_change(self, event):
        self._layer_selection_widget.points_layer.choices = (
            self._get_valid_points_layers(None)
        )
        self._layer_selection_widget.image_layer.choices = (
            self._get_valid_image_layers(None)
        )
        self._layer_selection_widget_mask.image_layer.choices = (
            self._get_valid_image_layers(None)
        )
        self._layer_selection_widget_mask.mask_layer.choices = (
            self._get_valid_mask_layers(None)
        )
        self._layer_selection_widget_connected_components.mask_layer.choices = self._get_valid_image_layers(
            None
        )
        self._layer_selection_widget_display_connected_components.components_layer.choices = self._get_valid_labels_layers(
            None
        )
        self.store_tomogram_widget.image_layer.choices = (
            self._get_valid_labels_layers(None)
        )
        self.store_all_components_widget.image_layer.choices = (
            self._get_valid_labels_layers(None)
        )
        self.color_distances_widget.image_layer.choices = (
            self._get_valid_labels_layers(None)
        )

    def _on_click_color_point(self):
        """
        This is to select a point in the foreground (i.e. non-zero voxels) in 3D.
        """
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_click)

    def _on_mouse_click(self, viewer, event):
        coordinates = None
        if (
            event.position is not None
            and self.color_distances_widget.image_layer.value is not None
        ):
            status = self.color_distances_widget.image_layer.value.get_status(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            intersect_coords = status["coordinates"]
            coordinates = np.array(
                list(
                    map(
                        int,
                        intersect_coords.split(":")[0].strip(" []").split(),
                    )
                )
            )
            value = int(intersect_coords.split(":")[1])
            if value != 0:
                self.color_distances_widget.point.value = coordinates
                self.viewer.mouse_drag_callbacks.remove(self._on_mouse_click)

    def _on_click_freehand(self):
        """
        This is for freehand drawing of the polygon.

        This is a preliminary implementation and is not perfect, particularly the visual feedback.
        """

        # make sure that we are in 3D view
        if self.viewer.dims.ndisplay != 3:
            napari.utils.notifications.show_warning("Please switch to 3D view")
            return

        # Initialize a dummy shape layer to get the dimensions right (3D)
        shape_layer = self.viewer.add_shapes(
            np.ones((2, 3)),
            shape_type="polygon",
            edge_color="coral",
            face_color="royalblue",
            opacity=0.5,
            name="lasso-shapes",
        )

        # Add a callback to the shape layer, imitating the 2D lasso tool
        @shape_layer.mouse_drag_callbacks.append
        def add_polygon(shape_layer, event):
            # Disable camera interaction
            self.viewer.camera.interactive = False

            # Overwrite the get_value and edit methods of the shape layer (causing problems)
            shape_layer, original_get_value, original_edit = (
                redefine_shapelayer_functions(shape_layer)
            )

            # Set the mode to ADD_POLYGON_LASSO
            shape_layer._mode = Mode.ADD_POLYGON_LASSO

            # Activate drawing mode
            generator = add_path_polygon_lasso(shape_layer, event)
            yield
            while True:
                try:
                    if event.type == "mouse_move":
                        next(generator)
                        yield
                    elif event.type == "mouse_release":
                        next(generator)
                        yield
                        generator.close()
                        break
                except StopIteration:
                    # Get points in the correct order
                    points = np.concatenate(
                        (
                            shape_layer.data[1][:2],
                            shape_layer.data[0][2:],
                            shape_layer.data[0][1:2],
                        ),
                        axis=0,
                    )

                    # Add the points to the viewer
                    self.viewer.add_points(
                        points,
                        name="lasso-points",
                        edge_color="blue",
                        face_color="blue",
                        size=2,
                    )

                    # Remove the shape layer
                    self.viewer.layers.remove("lasso-shapes")

                    # Enable camera interaction
                    self.viewer.camera.interactive = True
                    break

    def _on_click_polygon(self):
        """
        This is for manually clicking each point of the polygon.
        """
        # make sure that we are in 3D view
        if self.viewer.dims.ndisplay != 3:
            self.viewer.dims.ndisplay = 3
        # initialize a points layer
        self.viewer.add_points(
            ndim=3,
            name="lasso-points",
            edge_color="blue",
            face_color="blue",
            size=3,
        )

    def _lasso_from_polygon(
        self,
        points_layer: napari.layers.Points,
        image_layer: napari.layers.Image,
    ):
        if (points_layer is None) or (image_layer is None):
            return

        # Get the selected points
        points = points_layer.data

        # get the volume shape
        volume_shape = image_layer.data.shape

        # generate the mask
        mask = mask_via_extension(points, volume_shape)

        # add the mask to the viewer
        mask_layer = self.viewer.add_image(mask, name="mask", opacity=0.4)
        mask_layer.colormap = "green"

        return

    def _color_distances(
        self,
        image_layer: napari.layers.Labels,
        point: List[int],
        connected_component_number: int,
        # starting_point: List[int] = None,
    ):
        if image_layer is None:
            return

        # get the volume
        foreground = image_layer.data == connected_component_number

        point = None
        if point is None:
            point = np.argwhere(foreground)[
                np.random.randint(np.sum(foreground))
            ]

        dist_field = dijkstra3d.euclidean_distance_field(
            foreground,
            source=[point],
        )

        dist_field[foreground == 0] = 0
        dist_field[np.isinf(dist_field)] = (
            0  # this should not happen, but just in case
        )
        dist_field = dist_field / np.max(dist_field)

        max_range = 50
        # discretize dist_field (i.e. make integers)
        dist_field = np.round(dist_field * max_range).astype(int)

        from matplotlib import pyplot as plt
        from napari.utils import CyclicLabelColormap

        cmap = plt.get_cmap("seismic")
        colors = cmap(np.linspace(0, 1, max_range + 1))
        colormap = CyclicLabelColormap(colors)

        # add to viewer with rainbox color map
        self.viewer.add_labels(
            dist_field, name="distance field", colormap=colormap
        )

    def _mask_volume(
        self,
        image_layer: napari.layers.Image,
        mask_layer: napari.layers.Image,
        masking: str,
    ):
        if (image_layer is None) or (mask_layer is None):
            return

        # get the mask
        mask = mask_layer.data

        # get the volume
        volume = image_layer.data

        masked_volume = volume.copy()
        if masking == "isolate":
            masked_volume[~mask] = 0
        elif masking == "subtract":
            masked_volume[mask] = 0

        # add the masked volume to the viewer
        self.viewer.add_image(masked_volume, name="masked_volume")

    def _connected_components(
        self,
        mask_layer: napari.layers.Image,
        remove_small_objects_size: int,
    ):
        if mask_layer is None:
            return

        mask = mask_layer.data
        mask = mask > 0

        # get the connected components
        components, num_components = label(mask)

        # remove small objects
        max_val = np.max(components)
        i = 1
        while i < max_val + 1:
            if np.sum(components == i) < remove_small_objects_size:
                components[components == i] = 0
                components[components > i] -= 1
                max_val -= 1
            else:
                i += 1

        # add as labels layer
        self.viewer.add_labels(components, name="connected_components")

    def _display_connected_components(
        self,
        components_layer: napari.layers.Labels,
        component_number: int,
    ):
        if components_layer is None:
            return

        max_label = components_layer.data.max()
        colors = {i: (0, 0, 0, 0) for i in range(max_label + 1)}
        colors[component_number] = (
            1,
            0,
            0,
            1,
        )  # Set the label of interest to red with full opacity

        # Apply the custom colormap to the existing layer
        components_layer.color = colors

    def _store_tomogram(
        self,
        image_layer: napari.layers.Image,
        store_component_number: int,
        filename: str,
    ):
        if image_layer is None:
            return
        out_data = (image_layer.data == store_component_number) * 1.0
        out_data = np.transpose(out_data, (2, 1, 0))
        store_tomogram(filename, out_data)

    def _store_all_components(
        self,
        image_layer: napari.layers.Image,
        foldername: str,
    ):
        print("Storing all components")
        if image_layer is None:
            return
        out_data = image_layer.data
        for i in range(1, np.max(out_data) + 1):
            print("Storing component", i)
            out_data_i = (out_data == i) * 1.0
            out_data_i = np.transpose(out_data_i, (2, 1, 0))
            store_tomogram(str(foldername) + f"/component_{i}.mrc", out_data_i)

    def _get_valid_points_layers(
        self, combo_box
    ) -> List[napari.layers.Points]:
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Points)
        ]

    def _get_valid_image_layers(self, combo_box) -> List[napari.layers.Image]:
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

    def _get_valid_labels_layers(
        self, combo_box
    ) -> List[napari.layers.Labels]:
        return [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _get_valid_mask_layers(self, combo_box) -> List[napari.layers.Image]:
        image_layers = self._get_valid_image_layers(combo_box)
        # only return binary images
        return [layer for layer in image_layers if layer.data.dtype == bool]

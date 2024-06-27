def redefine_shapelayer_functions(shape_layer):
    """
    Redefine get_value method and edit method of a shape layer.
    These were causing issues with converting the lasso tool to 3D.

    I guess this is just a temporary fix until we can figure out how to properly implement the lasso tool in 3D.
    """
    # Store the original get_value method
    original_get_value = shape_layer.get_value
    original_edit = shape_layer._data_view.edit

    # Define a new get_value method
    def custom_get_value(
        self,
        position,
        *,
        view_direction=None,
        dims_displayed=None,
        world=False,
    ):
        # Simply return ones -- I think this leads to the bad visualizations but doesnt change the functionality
        self._value = (0, 0)
        return (0, 0)

    def custom_edit(
        self,
        index,
        data,
        face_color=None,
        edge_color=None,
        new_type=None,
    ):
        # set new_type to None
        original_edit(
            index,
            data,
            face_color=face_color,
            edge_color=edge_color,
            new_type=None,
        )

    # Overwrite the get_value method for the specific instance
    shape_layer.get_value = custom_get_value.__get__(
        shape_layer, type(shape_layer)
    )
    shape_layer._data_view.edit = custom_edit.__get__(
        shape_layer, type(shape_layer)
    )

    return shape_layer, original_get_value, original_edit

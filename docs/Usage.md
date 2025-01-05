## How to use the lasso plugin

### 1. Preparation
- Open the napari viewer
- Load a 3D image (binary)
- Load the lasso plugin

### 2. Draw Lasso
First, click the "Freehand" button in the upper right corner of the viewer. Then, draw a lasso on the image by clicking and dragging the mouse. 

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/48f24e49-4cd0-4fa1-9f22-bfe8f9ae09bb" alt="lasso_selection" width="49%" />
    <img src="https://github.com/user-attachments/assets/bf73433d-f9f2-4f7d-b798-de2c71f3197a" alt="lasso_draw" width="49%" />
</div>


### 3. Generate mask and mask out the image
Click the "Lasso" button to generate a mask in the shape of the image you want to mask. Then click the "Mask Volume" button to mask out the image and generate a new layer with the masked image ("masked_volume").


<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/98242f8a-bfa7-44ed-9a44-aed49c124955" alt="lasso_mask_generation" width="49%" />
    <img src="https://github.com/user-attachments/assets/8dff4d4d-c0ce-4c31-abc5-f730c69f62c8" alt="lasso_masked_volume" width="49%" />
</div>


### 4. Compute connected components
By selecting the "masked_volume" layer and clicking the "Connected Components" button, you can compute the connected components of the masked image. The connected components will be displayed as a new layer ("connected_components").

You can choose the remove small objects from the components by setting the "remove small objects" parameter to something other than 0. This will remove all connected components with a volume smaller than the specified value.

You also the the option to perform morphological opening before computing the connected components. This could be useful to split components that were wrongly merged by single voxels in the initial segmentations.

#### Visualization
To look at a single connected component, you can select the number of the component you would like to visualize and click "Display Connected Components" to display the selected component. All others will be blacked out.

If you would like to display all components again, you can select component number 0 and click "Display Connected Components".

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/2e4571ee-ab4c-4c01-bcfe-df3ceb2f9a60" alt="lasso_compute_components" width="49%" />
    <img src="https://github.com/user-attachments/assets/17253256-258e-4861-99d5-07c32f47bc07" alt="lasso_visualize_single_membrane" width="49%" />
</div>

### 5. Save out the connected components
You can now save out the components you would like to keep by selecting the corresponding component number, specifiying a file path, and clicking the "Store Tomogram" button. This will save the selected component as a new .mrc file.

Alternatively, you also also specify a directory path and select "Store All Components" to save out all components as individual .mrc files.

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/14c8b195-f439-49c4-8d9e-6af6c80c82eb" alt="lasso_store_all_comps" width="49%" />
    <img src="https://github.com/user-attachments/assets/126cab19-3c36-4556-b674-ed79d5deaa18" alt="generated_files" width="49%" />
</div>
Skull extraction
================

Binary segmentation mask to mesh
--------------------------------
Run the following command to generate mesh (.vtk).

.. code-block:: shell

    PATH_TO_3DSLICER_EXECUTABLE_FILE --no-splash --no-main-window --python-script PATH_TO/mask_to_mesh.py


Skull mesh extraction from brain mesh
-------------------------------------
Extract skull mesh (.vtk) and load the saved .vtk file in 3D Slicer to visualize it along with binary brain mask. Script has base workflow to extract skull, which can be improved as needed.

.. code-block:: shell

    python process_skull.py

Export skull mesh to its segmentation
-------------------------------------
Save skull segmentation as .nrrd file.

.. code-block:: shell

    PATH_TO_3DSLICER_EXECUTABLE_FILE --no-splash --no-main-window --python-script PATH_TO/save_skull_data.py
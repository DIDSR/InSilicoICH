Generate LabelMap from Markups
==============================

Skull suture annotation
-----------------------
Skull sutures annotated for `NIHPD_Head <https://github.com/DIDSR/PedSilicoICH/blob/a8674c7446cf7ad700feb387847e11540b0c4f93/src/pedsilicoICH/ground_truth_definition/phantoms.py#L569>`_ in `markups <https://github.com/DIDSR/PedSilicoICH/tree/5640c3a5bc5a5473d373db38ef62652bb47260a7/src/pedsilicoICH/annotations/suture/NIHPD_Head_Phantom/markups>`_:

- Saggital suture (``saggital``)
- Coronal suture (``coronal_l`` and ``coronal_r``)
- Lambdoid suture (``lambdoid_l`` and ``lambdoid_r``)
- Squamosal suture (``squamosal_l`` and ``squamosal_r``)

Skull sutures are annotated using Markups tool from 3D Slicer. Markups can be manually converted into Models followed by Segmentation/LabelMap using 3D Slicer extension `Slicer Markups to Model <https://github.com/SlicerIGT/SlicerMarkupsToModel>`_, however, to avoid manual process, LabelMap (segmentation labels) can be created with the script below.

To install Markups to Model go to View/Extensions Manager then search for Markups to Model and install, then run the following:

Code execution
--------------
Use the installed 3D Slicer executable file to run the script.

Note: The NIHPD Phantom is required to be downloaded as mentioned in `NIHPD_Head <https://github.com/DIDSR/PedSilicoICH/blob/a8674c7446cf7ad700feb387847e11540b0c4f93/src/pedsilicoICH/ground_truth_definition/phantoms.py#L569>`_.

.. code-block:: shell

    PATH_TO_3DSLICER_EXECUTABLE_FILE --no-splash --no-main-window --python-script PATH_TO/markup_to_label.py
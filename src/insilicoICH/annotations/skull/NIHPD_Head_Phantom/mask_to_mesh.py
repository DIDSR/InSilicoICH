"""
Create mesh file from segmentation in NIfTI.

Note: run 3D Slicer command from the current directory. 
"""
import os
import slicer

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))

if not os.path.exists(os.path.join(main_directory, "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets")):
    os.makedirs(os.path.join(main_directory, "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets"))

# Path to brain mask (NIfTI)
path_mask_brain = os.path.join(
        main_directory,
        "src/NIHPD_Head_Phantom",
        "nihpd_asym_04.5-08.5_mask.nii"
    )

# Path to save brain mesh (VTK)
path_vtk_brain = os.path.join(
        main_directory,
        "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
        "mesh_brain.vtk",
    )

# Load the segmentation file
segmentationNode = slicer.util.loadSegmentation(path_mask_brain)

# Export to model
shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), 'exported_model')
slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segmentationNode, exportFolderItemId)

# Save as VTK
modelNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLModelNode')
slicer.util.saveNode(modelNode, path_vtk_brain)
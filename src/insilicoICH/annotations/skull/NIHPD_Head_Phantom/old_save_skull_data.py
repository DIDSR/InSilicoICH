"""
Export skull mesh into its segmentation.
"""

import os
import sys
import slicer
from skull import Skull

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


class SaveSkullData(Skull):
    def __init__(self):
        super().__init__()

    def load_mesh(self, path_mesh: str):
        print(os.path.exists(path_mesh))
        self.mesh_skull = slicer.util.loadModel(path_mesh)

    def generate_skull_segmentation(
        self, path_reference_geometry_nifti: str, path_save_segmentation: str
    ) -> None:
        """
        Generate segmentation (as LabelMap) for skull.
        Note: current export is closed surface, it should be open surface.

        Args:
            path_reference_geometry_nifti (str): Path to the reference NIfTI image.
            path_save_segmentation (str): Path to the output NRRD label file.
        """
        mesh_node = self.mesh_skull

        # Create model node
        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        model_node.SetAndObservePolyData(mesh_node.GetPolyData())

        # Create segmentation node
        segmentation_node = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(segmentation_node)
        segmentation_node.CreateDefaultDisplayNodes()

        # Import model to segmentation
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(
            model_node, segmentation_node
        )

        # Load reference volume
        reference_volume = slicer.util.loadVolume(path_reference_geometry_nifti)

        # Set reference geometry
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(
            reference_volume
        )

        # Create segment
        segmentation = segmentation_node.GetSegmentation()
        segmentation.AddEmptySegment("skull")

        # Create labelmap volume node
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

        # Export segmentation to labelmap
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentation_node, labelmap_node
        )

        # Save labelmap
        slicer.util.saveNode(labelmap_node, path_save_segmentation)


if __name__ == "__main__":
    object_skull_data = SaveSkullData()
    object_skull_data.load_mesh(
        path_mesh=os.path.join(
            main_directory,
            "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
            "mesh_skull.vtk",
        )
    )

    object_skull_data.generate_skull_segmentation(
        path_reference_geometry_nifti=os.path.join(
            main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
        ),
        path_save_segmentation=os.path.join(
            main_directory,
            "src/pedsilicoICH/annotations/skull/NIHPD_Head_Phantom/assets",
            "segmentation_skull.nrrd",
        ),
    )

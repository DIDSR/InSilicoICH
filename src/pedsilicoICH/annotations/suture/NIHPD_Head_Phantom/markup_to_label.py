"""
Create segmentation label (LabelMap) from annotated markups.

The script is supposed to be used with 3D Slicer as mentioned in the documentation.
"""

import slicer
import os
import sys

main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.append(main_directory)


def convert_multiple_markups_to_segmentation_label(
    markup_json_dir: str, reference_nifti_path: str, output_path: str
):
    """
    Create label (segmentation) from annotated markups for the given NIfTI image.

    Args:
        markup_json_dir (str): Directory with created markup JSON files.
        reference_nifti_path (str): Path to the reference NIfTI image.
        output_path (str): Path to the output NRRD label file.
    """
    # Load reference volume
    reference_volume = slicer.util.loadVolume(reference_nifti_path)

    # Create segmentation node
    segmentation_node = slicer.vtkMRMLSegmentationNode()
    slicer.mrmlScene.AddNode(segmentation_node)
    segmentation_node.CreateDefaultDisplayNodes()

    # Set reference geometry
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(reference_volume)

    # Counter for sequential labeling
    segment_counter = 0

    # Process each markup file
    for markup_file in os.listdir(markup_json_dir):
        if markup_file.endswith(".mrk.json"):
            markup_path = os.path.join(markup_json_dir, markup_file)
            segment_name = os.path.splitext(os.path.splitext(markup_file)[0])[0]

            # Load markup
            markup_node = slicer.util.loadMarkups(markup_path)

            # Create model node
            model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")

            # Create MarkupsToModel node
            markupsToModel = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsToModelNode"
            )
            markupsToModel.SetAndObserveInputNodeID(markup_node.GetID())
            markupsToModel.SetAndObserveOutputModelNodeID(model_node.GetID())

            # Set curve parameters
            markupsToModel.SetModelType(slicer.vtkMRMLMarkupsToModelNode.Curve)
            markupsToModel.SetCurveType(slicer.vtkMRMLMarkupsToModelNode.CardinalSpline)
            markupsToModel.SetTubeRadius(1.0)
            markupsToModel.SetTubeLoop(False)
            markupsToModel.SetTubeNumberOfSides(8)
            markupsToModel.SetAutoUpdateOutput(True)

            # Import model to segmentation
            slicer.modules.segmentations.logic().ImportModelToSegmentationNode(
                model_node, segmentation_node
            )

            # Get the segment and set its name
            segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(
                segment_counter
            )
            segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
            segment.SetName(segment_name)

            segment_counter += 1

            # Cleanup individual nodes
            slicer.mrmlScene.RemoveNode(model_node)
            slicer.mrmlScene.RemoveNode(markupsToModel)
            slicer.mrmlScene.RemoveNode(markup_node)

    # Create labelmap volume node
    labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

    # Export segmentation to labelmap
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
        segmentation_node, labelmap_node
    )

    # Save labelmap
    slicer.util.saveNode(labelmap_node, output_path)

    # Final cleanup
    slicer.mrmlScene.RemoveNode(segmentation_node)
    slicer.mrmlScene.RemoveNode(reference_volume)
    slicer.mrmlScene.RemoveNode(labelmap_node)


if __name__ == "__main__":
    reference_nifti_path = os.path.join(
        main_directory, "src/NIHPD_Head_Phantom", "nihpd_asym_04.5-08.5_mask.nii"
    )
    output_nifti_path = os.path.join(
        main_directory,
        "src/pedsilicoICH/annotations/suture/NIHPD_Head_Phantom",
        "labelmap.nrrd",
    )

    markup_dir = os.path.join(
        main_directory, "src/pedsilicoICH/annotations/suture/NIHPD_Head_Phantom/markups"
    )

    convert_multiple_markups_to_segmentation_label(
        markup_dir, reference_nifti_path, output_nifti_path
    )

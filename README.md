# InSilicoICH: Synthetic Intracranial Hemorrhage Modeling Tools
<a href="https://github.com/DIDSR/InSilicoICH/actions/workflows/python-app.yml"><img src="https://github.com/DIDSR/InSilicoICH/actions/workflows/python-app.yml/badge.svg?branch=master" alt="Package Build and Testing Status"></a>

<p align="center">
  <img src="assets/InSilicoICH.png" alt="InSilicoICH Logo" width="400">
</p>
<p align="center">
  <img src="assets/pipeline.png" alt="Simulation Pipeline Diagram" width="800">
</p>

This repository contains tools for generating synthetic non-contrast CT datasets of **intracranial hemorrhage (ICH)**. These datasets are designed to be used for developing, testing, and evaluating AI detection devices.

## The Challenge: Data Scarcity in Medical AI

Intracranial hemorrhage (ICH) is a life-threatening brain bleed requiring immediate medical care. AI-powered Computer-Aided Triage (CADt) devices can accelerate diagnosis by analyzing emergency room CT scans (e.g., [Rapid ICH K221456](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K221456)).

However, a significant data gap exists for pediatric patients. ICH occurs less frequently in children, leading to a scarcity of annotated pediatric data. An AI model trained predominantly on adults may perform poorly on pediatric scans, potentially delaying time-sensitive treatment for children.

## Our Solution: In Silico Data Generation

To address this data availability challenge, **InSilicoICH** supplements real patient data with *in silico* (computer-simulated) data. By using realistic computational human phantoms and physics-based CT simulations, we can generate perfectly-labeled datasets at a fraction of the cost and time required for manual annotation of real data.

## How It Works

Our simulation pipeline combines several state-of-the-art components:

1.  **Digital Phantoms:** We use the MIDA phantom[^1] and the NIHPD head phantoms [^2][^3], which are detailed, anatomically-realistic patient-based models of the head.
2.  **Hemorrhage Insertion:** A knowledge-based algorithm inserts synthetic hemorrhages into the phantoms. This algorithm controls the placement, shape, volume, and attenuation based on models from [real hemorrhage segmentation data](https://arxiv.org/abs/2308.11298).
3.  **Physics-Based CT Simulation:** The final phantom, complete with the synthetic hemorrhage, is imaged using [**XCIST**](https://github.com/xcist/main), a realistic X-ray CT simulation framework that models the entire image acquisition process.

The tool currently supports three major hemorrhage subtypes:
* **Intraparenchymal (IPH)**
* **Epidural (EDH)**
* **Subdural (SDH)**

Intraventricular (IVH) and subarachnoid (SAH) hemorrhages are underdevelopment [here](https://github.com/DIDSR/InSilicoICH/tree/sah-ivh).

## Controllable Parameters

The simulation is highly customizable, allowing you to control dozens of parameters.

* **Patient Characteristics**
    * **Phantom Identifier**: Select a specific phantom from the cohort.
    * **Age**: `6.5 - 38 years` (based on the available phantom library).

* **Lesion Characteristics**
    * **Hemorrhage Type**: `IPH`, `SDH`, `EDH`, or `None`.
    * **Hemorrhage Volume**: `0 - 100 mL`.
    * **Intensity (Attenuation)**: `-30 - 100 HU`.
    * **Mass Effect**: `True` or `False`.
    * **Edema**: `0 - 15 voxels` (IPH only).

* **Acquisition Characteristics**
    * **X-ray Tube Current**: `10 - 1500 mA`.
    * **X-ray Tube Peak Voltage**: `70 - 140 kVp`.
    * **Reconstruction FOV**: `100 - 500 mm`.
    * **Reconstruction Kernel**: e.g., `Soft`, `Standard`, `Bone`.
    * **View Count**: Number of projection views (e.g., `1000`).

* **Output & Metadata**
    * **Random Seed**: For reproducibility.
    * **Image & Mask Locations**: Control where output files are saved.

## Example Simulations

Below are example outputs using the MIDA phantom, showing the simulated CT images and the corresponding ground truth segmentation masks.

**Simulated CT Images with ICH**
<p align="center">
  <img src="assets/MIDA_montage_noMask.png" alt="Montage of simulated CT scans with various ICH types" width="800">
</p>

**Ground Truth Segmentation Masks**
<p align="center">
  <img src="assets/MIDA_montage_Mask.png" alt="Montage of ground truth segmentation masks for the ICH" width="800">
</p>

## Tutorials and Usage

For worked examples and tutorials on how to use the included functions and tools to generate phantoms with ICH and all relevant parameters, please refer to the `notebooks` folder.

---

## Installation and Setup

### 1. Prerequisites
* Python (tested on versions >=3.11 and <3.13)
* We recommend using a `conda` or `venv` virtual environment.

### 2. Install the Package

**Install Directly**

Install the package directly from GitHub using `pip`:

```bash
pip install git+https://github.com/DIDSR/InSilicoICH.git
```

**Install from GitHub Clone**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DIDSR/InSilicoICH.git
    cd InSilicoICH
    ```

2.  **Install dependencies:**
    ```bash
    pip install .
    ```

## Disclaimer
### About the Catalog of Regulatory Science Tools
The enclosed tool is part of the [Catalog of Regulatory Science Tools](https://cdrh-rst.fda.gov/), which provides a peer-reviewed resource for stakeholders to use where standards and qualified Medical Device Development Tools (MDDTs) do not yet exist. These tools do not replace FDA-recognized standards or MDDTs. This catalog collates a variety of regulatory science tools that the FDA’s Center for Devices and Radiological Health’s (CDRH) Office of Science and Engineering Labs (OSEL) developed. These tools use the most innovative science to support medical device development and patient access to safe and effective medical devices. If you are considering using a tool from this catalog in your marketing submissions, note that these tools have not been qualified as [Medical Device Development Tools](https://www.fda.gov/medical-devices/medical-device-development-tools-mddt) and the FDA has not evaluated the suitability of these tools within any specific context of use. You may [request feedback or meetings for medical device submissions](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program) as part of the Q-Submission Program.
For more information about the Catalog of Regulatory Science Tools, email [RST_CDRH@fda.hhs.gov](mailto:RST_CDRH@fda.hhs.gov).

## References
[^1]: U.S. Food and Drug Administration. (2023). MIDA: A Multimodal Imaging-Based Model of the Human Head and Neck (RST24NO05.01). <https://cdrh-rst.fda.gov/mida-multimodal-imaging-based-model-human-head-and-neck>
[^2]: VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins and BDCG, Unbiased average age-appropriate atlases for pediatric studies, NeuroImage, In Press, ISSN 1053–8119, DOI: 10.1016/j.neuroimage.2010.07.033
[^3]:	VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins Unbiased nonlinear average age-appropriate brain templates from birth to adulthood NeuroImage, Volume 47, Supplement 1, July 2009, Page S102 Organization for Human Brain Mapping 2009 Annual Meeting, DOI: 10.1016/S1053-8119(09)70884-5

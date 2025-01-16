CT Imaging Datasets for Pediatric Device Assessment of Intracranial Hemorrhage
==============================================================================

|tests|

.. image:: assets/project_aims.png
        :width: 800
        :align: center

.. |tests| image:: https://github.com/brandonjnelsonFDA/PedSilicoICH/actions/workflows/python-app.yml/badge.svg?branch=master
    :alt: Package Build and Testing Status
    :scale: 100%
    :target: https://github.com/brandonjnelsonFDA/PedSilicoICH/actions/workflows/python-app.yml

This repository contains tools for generating synthetic non contrast CT datasets of intracranial hemorrhage (ICH).

Motivation
----------

Computer aided triaging (CADt) devices for intracranial hemorrhage (ICH) in the emergency room (e.g. `Rapid ICH K221456 <https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K221456>`_) is one important example where pediatric and adult cases exist in a reading queue where pediatric patients could be disadvantaged by being deprioritized for time sensitive treatment using an adult-trained AI model that extrapolates poorly to pediatric patients. While these AI/ML devices have potential to benefit pediatric patients, there is currently a lack of annotated pediatric data for evaluating the balance of risk and benefits.

Purpose
-------

To address data availability challenges, we propose to supplement available pediatric patient computed tomography (CT) datasets with data generated in silico, generated using realistic computational human models and physics-based CT simulations. In silico data generation allows for creating examples with true labels with a fraction of the cost that is needed to label real patient data.

Methods
-------

We have previously combined the `pediatric and adult digital XCAT cohort of phantoms <https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.3480985>`_ with the `XCIST x-ray CT simulation framework <https://iopscience.iop.org/article/10.1088/1361-6560/ac9174/meta>`_ to create realistic CT exams. This preliminary work was in support of investigating the `effectiveness of deep learning denoising algorithms in pediatric patients <https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16901>`_.

In this work, synthetic hemorrhages are inserted into head and neck phantoms based on MR templates and atlases. Presently, three hemorrhage subtypes are supported: intraparenchymal (IPH), epidural (EDH), and subdural (SDH). A knowledge-based algorithm is used to guide the placement and shape of the synthetic hemorrhages, using volume and attenuation parameters modeled from `real hemorrhages obtained in a segmentation dataset <https://arxiv.org/abs/2308.11298>`_. Appropriate Hounsfield units can be assigned to each segmented region of the phantom, such as the gray and white matter, bone, CSF, and the hemorrhage. As with previous work, XCIST was used to create realistic simulated CT exams with included synthetic hemorrhages.

The knowledge-based algorithm allows the following parameters to be controlled:

+----------------------------+------------------------------------------------------+-------------------------------------------+---------------------------------+
|                            |                                                      |                                           |                                 |
| Patient Characteristics    | Lesion Characteristics                               | Acquisition Characteristics               | Misc./Output Data               |
+============================+======================================================+===========================================+=================================+
|                            |                                                      |                                           |                                 |
| Identifier                 | Intensity [HU]                                       | X-ray tube current [mA]                   | Seed to reproduce               |
+----------------------------+------------------------------------------------------+-------------------------------------------+---------------------------------+
|                            |                                                      |                                           |                                 |
| Age (atlas-based)          | Hemorrhage volume [mL] and slice coverage            | X-ray tube peak voltage [kVp]             | Image file location             |
+----------------------------+------------------------------------------------------+-------------------------------------------+---------------------------------+
|                            |                                                      |                                           |                                 |
|                            | Hemorrhage type                                      | CT acquisition view count [views]         | Mask file directory location    |
+----------------------------+------------------------------------------------------+-------------------------------------------+---------------------------------+
|                            |                                                      |                                           |                                 |
|                            | Mass effect strength (currently   IPH/round only)    | Reconstructed field of view (FoV) [mm]    | Hemorrhage slice number(s)      |
+----------------------------+------------------------------------------------------+-------------------------------------------+---------------------------------+
|                            |                                                      |                                           |                                 |
|                            | Edema [voxels] (IPH/round only)                      | Reconstruction kernel                     |                                 |
+----------------------------+------------------------------------------------------+-------------------------------------------+---------------------------------+

Below are example simulation outputs:

.. image:: assets/montage.png
        :width: 800
        :align: center

Installation
------------

.. code-block:: bash

        pip install git+https://github.com/DIDSR/PedSilicoICH.git

Tested on python 3.11.3

Usage
-----

The synthetic data generation and image simulation tools included in this repo can be used either programmatically by importing into Python scripts as a Package or via command line interface (CLI)

**Programmatic Usage**

See the included `jupyter notebooks <notebooks>`_ for example programmatic usage

**Command Line Usage**

After `pip` installing, the `pedsilicoich` program should be available in your environment and can be used as follows

.. code-block:: bash

        pedsilicoich example_config.toml

Any parameters provided in config files like `example_config.toml <example_config.toml>`_, override the `defaults <src/pedsilicoICH/configs/default.toml>`_.

Additionally, command line arguments can be provided as positional or keyword arguments, see the help string for more details:

.. code-block:: bash

        pedsilicoich --help

User provide command line arguments override user provided config files, which override the `default <src/pedsilicoICH/configs/default.toml>`_ configs

View a Sample Dataset (local demo)
----------------------------------

Based off of `the S-Synth Demo <https://github.com/DIDSR/ssynth-release>`_

.. image:: docs/assets/demo_preview.png
        :width: 800
        :align: center

A sample dataset is viewable as a demo, located in docs/index.html. To serve this website locally do the following:

1. install the `VS Code Live Server Extension <https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer>`_
2. Open `index.html <docs/index.html>`_ and click the `Go Live` button at the far lower right corner of VS Code that should appear when an html file is open. 
3. After selecting `Go`Under the `PORTS` tab of the VS Code terminal, add the port number that popped up after going live (5500 is default), then right click the forwarded address
4. Click on the `docs` folder containing the demo and the website should load

.. image:: docs/assets/live_server_help.png
        :width: 800
        :align: center

Module Layout
-------------

.. image:: assets/pedsilico_class_diagram.png
        :width: 800
        :align: center

Repository Contents
-------------------

**notebooks**: for introducing concepts, developing methods, scratch work, and running experiments

*Tutorials*

- `notebooks/tutorials/01_phantoms.ipynb <notebooks/tutorials/01_phantoms.ipynb>`_: introduce working with phantoms and lesion insertion to generate inputs for CT simulations.

- `notebooks/tutorials/02_scanners.ipynb <notebooks/tutorials/02_scanners.ipynb>`_: introduce working with virtual CT scanner for CT imaging simulations.

- `notebooks/tutorials/03_studies.ipynb <notebooks/tutorials/03_studies.ipynb>`_: integrates phantoms and scanners to run virtual imaging studies.

*Project Aims*

- `notebooks/00_basic_eda.ipynb <notebooks/00_basic_eda.ipynb>`_: exploratory data analysis of the Hssayeni et 2020 dataset [Aim 1.1]
- `notebooks/03_epidural_subdural_demo.ipynb <notebooks/03_epidural_subdural_demo.ipynb>`_: expand simulated lesions to subdural and epidural ICH [Aim 1.2]
- `notebooks/viewing_simulation_results.ipynb <notebooks/viewing_simulation_results.ipynb>`_: for viewing the simulation results from CT_dataset_pipeline.py
- `notebooks/IQ_evaluations.ipynb <notebooks/IQ_evaluations.ipynb>`_: basic evaluations for quality assurance of XCIST simulations and phantoms

**scripts**: for generating data sets and more production ready

- `CT_dataset_pipeline.py <CT_dataset_pipeline.py>`_: used for generating the in silico dataset

Contributing
------------

Our current practice is developing locally by cloning the git repo to your local machine or personal directory and accessing a common dataset. Commits are encouraged to be regularly synced between the local and remote repo. Contributions are welcome from all, though developing on your own branch may be best to avoid merge conflicts, then we can decide on what to merge to the main branch (see `software carpentry on collaborating with git <https://swcarpentry.github.io/git-novice/08-collab.html>`_ for details).

See Also
--------

- `PedSilicoAbdomen <https://github.com/DIDSR/PedSilicoAbdomen>`_ for generating synthetic abdominal non contrast CT datasets
- `PedSilicoLVO <https://github.com/brandonjnelsonFDA/PedSilicoLVO>`_ for generating synthetic large vessel occlusion (LVO) non contrast CT datasets
- `Virtual Imaging Tools (VITools) <https://github.com/DIDsr/vitools>`_ tools for running virtual imaging trials including image acquisition frameworks
- `Wait time assessments for ICH CADt VIT <https://github.com/brandonjnelsonFDA/ICH-CADt-VIT>`_

Disclaimer
----------

This software and documentation (the "Software") were developed at the **US Food and Drug Administration** (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

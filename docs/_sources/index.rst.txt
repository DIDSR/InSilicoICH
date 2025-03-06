Synthetic Intracranial Hemorrhage Modeling Tools
================================================

This documentation provides information regarding how to download, install, and use the Synthetic Intracranial Hemorrhage Modeling Tools.

Motivation
----------

Intracranial hemorrhage (ICH) is a bleeding in the brain that can result from trauma or stroke, it can be a life threatening condition that needs immediate care. Computer aided triaging (CADt) devices read CT scans taken in the emergency room to detect ICH (e.g. `Rapid ICH K221456 <https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K221456>`_). Both adults and pediatrics can present with ICH, but with different frequencies. Due to this difference in frequency, pediatric patients could be disadvantaged by being deprioritized for time sensitive treatment using an adult-trained AI model that poorly extrapolates to pediatrics. While these AI/ML devices have potential to benefit pediatric patients, there is currently a lack of annotated pediatric data for evaluating the balance of risk and benefits.

Purpose
-------

To address data availability challenges, we propose to supplement available pediatric patient computed tomography (CT) datasets with data generated in silico, generated using realistic computational human models and physics-based CT simulations. In silico data generation allows for creating examples with true labels with a fraction of the cost that is needed to label real patient data.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   function_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

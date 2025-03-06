# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'InSilicoICH'
copyright = '2024, Jayse Weaver, Brandon Nelson'
author = 'Jayse Weaver, Brandon Nelson'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[2])
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autosummary',
              'sphinx_tabs.tabs',
              'sphinxcontrib.inkscapeconverter',
              'nbsphinx']
 #https://docs.readthedocs.io/en/stable/guides/jupyter.html

templates_path = ['_templates']
exclude_patterns = []
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output
latex_engine = 'xelatex'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_theme_options = {
    'logo': 'montage.png',
    'logo_name': 'true',
    'description': 'Digital Pediatric Image Quality Phantoms and Simulations for Evaluating CT Denoising Methods',
    'font_family': 'Helvetica',
    'head_font_family': 'Helvetica',
    'font_size': '12pt',
    'fixed_sidebar': 'true',
    'page_width': '1200px',
    'sidebar_width': '250px'
}

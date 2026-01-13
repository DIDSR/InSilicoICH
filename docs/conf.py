# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'InSilicoICH'
copyright = '2024, DIDSR'
author = 'Brandon Nelson, Jayse M. Weaver, Dhaval Kadia'
release = '0.5.12'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

autodoc_mock_imports = ["VITools", "pydicom", "SimpleITK", "nibabel", "skimage", "scipy", "requests", "noise", "tomli", "tomllib", "tomli_w", "dotenv"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'static_demo']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = ['static_demo']  # Copies contents of docs/static_demo to output root

# Navigation
html_context = {
    "display_github": True,
    "github_user": "DIDSR",
    "github_repo": "InSilicoICH",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

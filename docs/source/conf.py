# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
version = "0.1.0"

project = 'random-ml'
copyright = '2025, Rahul Goswami'
author = 'Rahul Goswami'
release = '0.1.0'

needs_sphinx = "1.8"

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
# # https://docs.readthedocs.io/en/latest/faq.html?highlight=environ#how-do-i-change-behavior-for-read-the-docs
# on_rtd = os.environ.get("READTHEDOCS", None) == "True"
#
# # If extensions (or modules to document with autodoc) are in another directory,
# # add these directories to sys.path here. If the directory is relative to the
# # documentation root, use os.path.abspath to make it absolute, like shown here.
# if on_rtd:
#     sys.path.insert(0, os.path.abspath(os.path.pardir))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
]


autosummary_generate = True
autodoc_default_options = {
    "members": None,
    "inherited-members": None,
}

master_doc = "index"

project = "random-ml"
current_year = 2025
copyright = f"2025-{current_year}, Rahul Goswami and contributors"

# Napoleon settings
napoleon_google_docstring = False
napoleon_include_init_with_doc = False

# HTML Theme
html_theme = "pydata_sphinx_theme"

# html_theme_options = {
#     "header_links_before_dropdown": 6,
#     "secondary_sidebar_items": ["page-toc"],
#     "navbar_end": ["theme-switcher.html", "navbar-github-links.html"],
#     "navbar_persistent": ["search-field.html"],
#     "navbar_align": "left",
#     "navigation_with_keys": False,
# }
#
# html_sidebars = {
#     "cite": [],
#     "contributing": [],
#     "install": [],
# }

html_title = f"random-ml {version}"

templates_path = ['_templates']
exclude_patterns = []
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object

    Adapted from scipy.
    """


    return  "https://github.com/yuvrajiro/random-ml/blob/main/random_ml/" + info['module'].replace('.', '/') + ".py"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_static_path = ['_static']

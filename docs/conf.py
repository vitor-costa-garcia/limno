"""Sphinx configuration."""

import limno

project = "limno"
author = "Vitor da Costa Garcia"
copyright = f"2026, {author}"
release = limno.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build"]

# The docstrings write code between single backticks, as in Markdown
default_role = "py:obj"

autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "furo"
html_title = "limno"

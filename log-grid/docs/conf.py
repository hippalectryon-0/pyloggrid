# sphinx-build -b html docs/ docs/build -a -j auto
import tomllib
from datetime import datetime

with open("../pyproject.toml", "rb") as f:
    toml_data = tomllib.load(f)["tool"]["poetry"]

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# noinspection PyUnresolvedReferences
project = toml_data["name"]
# noinspection PyShadowingBuiltins
copyright = f"2019-{datetime.now().year}, pyloggrid Developers"
# noinspection PyUnresolvedReferences
author = ", ".join(toml_data["authors"])
# noinspection PyUnresolvedReferences
release = toml_data["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # 'sphinx.ext.autosectionlabel',
    "myst_parser",
    "autoapi.extension",
]
todo_include_todos = True

extlinks = {
    "doi": ("https://dx.doi.org/%s", "doi:%s"),
}

templates_path = ["_templates"]
exclude_patterns = []

# Setup auto doc

autosummary_generate = True
numpydoc_class_members_toctree = True
numpydoc_show_class_members = False
autoapi_type = "python"
autoapi_dirs = ["../pyloggrid"]
add_module_names = False
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True


def skip_submodules(app, what, name, obj, skip, options):
    for el in [".logger", "DataExplorer.T", "DataExplorer.PlotFun", "DataExplorer.DrawFuncDict", "Grid.convolver_c"]:
        if name.endswith(el):
            skip = True
            break
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# html_static_path = ["static"]

html_theme = "sphinx_rtd_theme"
html_favicon = "static/img/PyLogGrid.svg"

# -- Extension configuration -------------------------------------------------


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

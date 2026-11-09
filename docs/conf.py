"""Config."""  # noqa: INP001
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Find the path to the source files we want to document, relative to the location of this file,
# convert it to an absolute path.
py_path = os.path.join(os.getcwd(), os.path.dirname(__file__), "../")
sys.path.insert(0, os.path.abspath(py_path))

project = "QCut"
copyright = "2024, Joonas Nivala"
author = "Joonas Nivala"

# The short X.Y version.
version = ""
# The full version, including alpha/beta/rc tags.
release = ""
try:
    from QCut import __version__ as version
except ImportError:
    pass
else:
    release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "myst_nb",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
    "private-members": False,
}

autosummary_generate = True
autosummary_ignore_patterns = []

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "jupyter_execute",
    "build",
    "test",
    ".venv",
    "docs-venv",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# The theory pages use \ket and \bra, which MathJax does not define by default.
mathjax3_config = {
    "tex": {
        "macros": {
            "ket": [r"\left|#1\right\rangle", 1],
            "bra": [r"\left\langle#1\right|", 1],
            "braket": [r"\left\langle#1\middle|#2\right\rangle", 2],
            "Tr": r"\operatorname{Tr}",
        }
    }
}

nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_logo = "_static/images/logos/qcut-logo-1024.png"
html_title = "QCut"
html_favicon = "_static/images/logos/qcut-logo.svg"
html_static_path = ["_static"]
html_theme_options = {"navbar_persistent": []}


def _public_bases(app, name, obj, options, bases):
    """List a private base's own bases in its place, since it is not documented."""
    public = []
    for base in bases:
        if getattr(base, "__name__", "").startswith("_"):
            _public_bases(app, name, obj, options, base_bases := list(base.__bases__))
            public.extend(base_bases)
        else:
            public.append(base)
    bases[:] = public


def _check_api_documented(app, env):
    """Warn about anything QCut exports that no API reference page documents."""
    import QCut
    import QCut.primitives
    from sphinx.util import logging

    documented = set(env.get_domain("py").objects)
    exported = [f"QCut.{name}" for name in QCut.__all__] + [
        f"QCut.primitives.{name}"
        for name in QCut.primitives.__all__
        if name not in QCut.__all__
    ]
    for name in exported:
        if name not in documented:
            logging.getLogger(__name__).warning(
                f"{name} is exported but not documented on any page under docs/api/"
            )


def setup(app):
    app.connect("autodoc-process-bases", _public_bases)
    app.connect("env-check-consistency", _check_api_documented)

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'GOP'
copyright = '2024, Индыков Дмитрий Андреевич'
author = 'Индыков Дмитрий Андреевич'
version = '2.0.0'
release = '2.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to a template name.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for autodoc -----------------------------------------------------

# This value selects what content will be inserted into the main body of an
# autoclass directive. The possible values are:
# 'class' - Only the class' docstring is inserted. This is the default.
# 'both' - Both the class' and the __init__ method's docstring are
# concatenated and inserted.
# 'init' - Only the __init__ method's docstring is inserted.
autoclass_content = 'both'

# This value selects if automatically documented members are sorted
# alphabetically (value 'alphabetical'), by member type (value 'groupwise')
# or by source order (value 'bysource'). The default is alphabetical.
autodoc_member_order = 'bysource'

# This value is a list of autodoc directive flags that should be automatically
# applied to all autodoc directives. The supported flags are:
# 'members', 'undoc-members', 'private-members', 'special-members',
# 'inherited-members' and 'show-inheritance'.
autodoc_default_flags = [
    'members',
    'undoc-members',
    'show-inheritance',
]

# -- Options for Napoleon -----------------------------------------------------

# Enable parsing of Google style docstrings
napoleon_google_docstring = True

# Enable parsing of NumPy style docstrings
napoleon_numpy_docstring = True

# Include init docstrings in class docstring
napoleon_include_init_with_doc = False

# Include private members (like _membername) with docstrings in the documentation
napoleon_include_private_with_doc = False

# Include special members (like __membername__) with docstrings in the documentation
napoleon_include_special_with_doc = True

# Use the .. admonition:: directive for the Example and Examples sections
napoleon_use_admonition_for_examples = False

# Use the .. admonition:: directive for the Note and Notes sections
napoleon_use_admonition_for_notes = False

# Use the .. admonition:: directive for the References and References sections
napoleon_use_admonition_for_references = False

# Use the :ivar: role for instance variables
napoleon_use_ivar = False

# Use the :param: role for function parameters
napoleon_use_param = True

# Use the :rtype: role for return values
napoleon_use_rtype = True

# Use the :keyword: role for keyword arguments
napoleon_use_keyword = True

# -- Options for intersphinx -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# -- Options for autosummary -------------------------------------------------

# Generate the autosummary pages when building
autosummary_generate = True

# -- Options for Russian language ---------------------------------------------

# Language for text automation
language = 'ru'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'GOP.tex', 'GOP Documentation',
     'Индыков Дмитрий Андреевич', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'gop', 'GOP Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'GOP', 'GOP Documentation',
     author, 'GOP', 'Гиперспектральная обработка и анализ растений',
     'Научные вычисления'),
]
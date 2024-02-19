author = "Allen Goodman"

autoapi_dirs = ["../src/yeji"]

copyright = "2024 Genentech, Inc"

exclude_patterns = [".DS_Store", "Thumbs.db", "_build"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"

project = "Yeji"

templates_path = ["_templates"]

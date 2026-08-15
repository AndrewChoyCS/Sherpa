"""JSON API wrapper around the EgoVerse curriculum pipeline.

Exists so the React frontend in ``web/`` can reach the same code the CLI and the
Streamlit dashboard run. Deliberately thin: no numeric logic lives here, every
endpoint delegates to :mod:`src`, so the browser cannot show a number that
``run_pipeline.py`` would not also write to ``reports/``.
"""

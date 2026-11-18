"""Make `pytest` work from a fresh clone, without an editable install.

pytest prepends the test file's own directory to sys.path, not the project
root, so `import constellation` would otherwise fail until someone ran
`pip install -e .`. Putting the root on the path here means the documented
command in the README works as written.
"""
import pathlib
import sys

ROOT = str(pathlib.Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

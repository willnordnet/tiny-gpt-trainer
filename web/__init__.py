"""An optional live viewer for a training run.

Nothing in `tinygpt/` imports anything from here. This package shells out to
`python -m tinygpt.train` and friends, reads their stdout, and serves the
result as a web page -- so the trainer stays a standalone command-line
project that happens to have a dashboard available, not one that needs it.
See DESIGN.md section 9, and web/README.md to run it.
"""

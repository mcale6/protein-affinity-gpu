import os


# tinygrad expects DEBUG to be an integer. Some local shells export
# DEBUG=release, which breaks module import during test collection.
if not os.environ.get("DEBUG", "0").isdigit():
    os.environ["DEBUG"] = "0"

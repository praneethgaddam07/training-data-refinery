"""
Storage backend abstraction: local paths and s3:// URIs both resolve through
pyarrow.fs, so pipeline stages can write/read either backend through the same
PyArrow calls. A path with no `s3://` prefix always resolves to local disk --
existing local behavior is unaffected by anything in this module.

The AWS variant is opt-in per-invocation: pass an `s3://bucket/prefix` path
instead of a local one (e.g. `shard_writer.py --shards-dir s3://my-bucket/shards`).
There is no hidden global switch -- what backend is used is always visible in
the path you pass.
"""

import os

import pyarrow.fs as fs


def is_s3(path: str) -> bool:
    return path.startswith("s3://")


def resolve(path: str):
    """Returns (filesystem, path_without_scheme) for a local path or s3:// URI."""
    if is_s3(path):
        if "region=" not in path:
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
            sep = "&" if "?" in path else "?"
            path = f"{path}{sep}region={region}"
    else:
        # from_uri() requires an absolute path -- relative local paths need resolving first
        path = os.path.abspath(path)
    return fs.FileSystem.from_uri(path)


def shards_exist(shards_dir: str) -> bool:
    filesystem, base_path = resolve(shards_dir)
    try:
        infos = filesystem.get_file_info(fs.FileSelector(base_path, recursive=True))
    except FileNotFoundError:
        return False
    return any(info.path.endswith(".parquet") for info in infos)


def open_shards_dataset(shards_dir: str, partitioning: str = "hive"):
    """pyarrow.dataset.dataset() over either a local shards dir or an s3:// prefix."""
    import pyarrow.dataset as ds

    filesystem, base_path = resolve(shards_dir)
    return ds.dataset(base_path, filesystem=filesystem, partitioning=partitioning)

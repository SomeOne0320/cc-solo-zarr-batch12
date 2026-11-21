# Working with groups

Zarr supports hierarchical organization of arrays via groups. As with arrays,
groups can be stored in memory, on disk, or via other storage systems that
support a similar interface.

To create a group, use the [`zarr.group`][] function:

```python exec="true" session="groups" source="above" result="ansi"
import zarr
store = zarr.storage.MemoryStore()
root = zarr.create_group(store=store)
print(root)
```

Groups have a similar API to the Group class from [h5py](https://www.h5py.org/). For example, groups can contain other groups:

```python exec="true" session="groups" source="above"
foo = root.create_group('foo')
bar = foo.create_group('bar')
```

Groups can also contain arrays, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
z1 = bar.create_array(name='baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
print(z1)
```

Members of a group can be accessed via the suffix notation, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root['foo'])
```

The '/' character can be used to access multiple levels of the hierarchy in one
call, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root['foo/bar'])
```

```python exec="true" session="groups" source="above" result="ansi"
print(root['foo/bar/baz'])
```

The [`zarr.Group.tree`][] method can be used to print a tree
representation of the hierarchy, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root.tree())
```

The [`zarr.open_group`][] function provides a convenient way to create or
re-open a group stored in a directory on the file-system, with sub-groups stored in
sub-directories, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
root = zarr.open_group('data/group.zarr', mode='w')
print(root)
```

```python exec="true" session="groups" source="above" result="ansi"
z = root.create_array(name='foo/bar/baz', shape=(10000, 10000), chunks=(1000, 1000), dtype='int32')
print(z)
```

For more information on groups see the [`zarr.Group` API docs](../api/zarr/group.md).

## Copying groups

A whole group -- including its sub-groups (even empty ones), arrays, attributes, and the
raw chunk data -- can be copied into another, already existing group with [`zarr.copy`][].
The copy works at the storage level, so the destination keeps the exact Zarr format and
storage layout of the source: Zarr format 2 and format 3 groups retain their respective
metadata documents and chunk key encodings. Only chunks that were actually written in the
source are copied, so regions that were never written stay sparse in the destination.

```python exec="true" session="copy" source="above" result="ansi"
import numpy as np
import zarr

source_store = zarr.storage.MemoryStore()
source = zarr.open_group(source_store, mode="w")
source.attrs["experiment"] = "copy demo"
source.create_group("measurements")  # an empty sub-group
array = source.create_array(
    "measurements/temperature",
    shape=(100,),
    chunks=(10,),
    dtype="float32",
    fill_value=np.nan,
)
array[: 20] = np.linspace(15.0, 25.0, 20)  # only two of the ten chunks are written

destination_store = zarr.storage.MemoryStore()
destination = zarr.open_group(destination_store, mode="w")

n_copied, n_skipped, n_bytes_copied = zarr.copy(source, destination)
print((n_copied, n_skipped, n_bytes_copied > 0))
```

Reopening the destination shows the same hierarchy, attributes, shapes, chunks, data types,
fill values and data, and the unwritten chunks are still unwritten:

```python exec="true" session="copy" source="above" result="ansi"
reopened = zarr.open_group(destination_store, mode="r")
print(reopened.attrs["experiment"])
print(list(reopened.keys()))
copied_array = reopened["measurements/temperature"]
print(copied_array.shape, copied_array.chunks, copied_array.dtype, copied_array.fill_value)
print(copied_array.nchunks_initialized, array.nchunks_initialized)
print(np.array_equal(copied_array[:20], array[:20]))
```

By default, [`zarr.copy`][] raises a [`zarr.errors.CopyError`][] when a node with the same
name already exists in the destination, and the check happens before anything is written,
so a failed copy never leaves a partially written destination. Pass `if_exists="skip"` to
leave the conflicting node -- including the whole contents of a conflicting sub-group --
untouched while copying every other branch. Skipped arrays are reported in the second
statistic:

```python exec="true" session="copy" source="above" result="ansi"
other_destination = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
other_destination.create_array("measurements/temperature", shape=(1,), dtype="float32")

n_copied, n_skipped, n_bytes_copied = zarr.copy(
    source, other_destination, if_exists="skip"
)
print((n_copied, n_skipped))
```

A dry run reports the same three statistics without writing anything to the destination;
the number of bytes copied is always zero in a dry run:

```python exec="true" session="copy" source="above" result="ansi"
dry_destination = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
print(zarr.copy(source, dry_destination, dry_run=True))
```

[`zarr.copy_store`][] is a store-oriented variant that opens the source and destination
groups from store-like objects at explicit paths, which is convenient for groups nested in
a store or stored in local directories:

```python exec="true" session="copy" source="above" result="ansi"
another_destination_store = zarr.storage.MemoryStore()
zarr.open_group(another_destination_store, mode="w")
stats = zarr.copy_store(source_store, another_destination_store)
print(stats[0], stats[1])
```

Copying is supported between in-memory ([`zarr.storage.MemoryStore`][]) stores and between
local directory ([`zarr.storage.LocalStore`][]) stores of the same type and Zarr format.
The source and destination must not overlap within the same store, and only the contents
of the source group are copied -- sibling groups elsewhere in the store are not affected.
Asynchronous equivalents are available as [`zarr.api.asynchronous.copy`][],
[`zarr.api.asynchronous.copy_all`][] and [`zarr.api.asynchronous.copy_store`][]; all entry
points return the same `(n_copied, n_skipped, n_bytes_copied)` tuple.

## Batch Group Creation

You can also create multiple groups concurrently with a single function call. [`zarr.create_hierarchy`][] takes
a [`zarr Storage instance`](../api/zarr/storage.md) instance and a dict of `key : metadata` pairs, parses that dict, and
writes metadata documents to storage:

```python exec="true" session="groups" source="above" result="ansi"
from zarr import create_hierarchy
from zarr.core.group import GroupMetadata
from zarr.storage import LocalStore

from pprint import pprint
import io

node_spec = {'a/b/c': GroupMetadata()}
nodes_created = dict(create_hierarchy(store=LocalStore(root='data'), nodes=node_spec))
# Report nodes (pprint is used for cleaner rendering in the docs)
output = io.StringIO()
pprint(nodes_created, stream=output, width=60)
print(output.getvalue())
```

Note that we only specified a single group named `a/b/c`, but 4 groups were created. These additional groups
were created to ensure that the desired node `a/b/c` is connected to the root group `''` by a sequence
of intermediate groups. [`zarr.create_hierarchy`][] normalizes the `nodes` keyword argument to
ensure that the resulting hierarchy is complete, i.e. all groups or arrays are connected to the root
of the hierarchy via intermediate groups.

Because [`zarr.create_hierarchy`][] concurrently creates metadata documents, it's more efficient
than repeated calls to [`create_group`][zarr.create_group] or [`create_array`][zarr.create_array], provided you can statically define
the metadata for the groups and arrays you want to create.

## Array and group diagnostics

Diagnostic information about arrays and groups is available via the `info`
property. E.g.:

```python exec="true" session="groups" source="above" result="ansi"
store = zarr.storage.MemoryStore()
root = zarr.group(store=store)
foo = root.create_group('foo')
bar = foo.create_array(name='bar', shape=1000000, chunks=100000, dtype='int64')
bar[:] = 42
baz = foo.create_array(name='baz', shape=(1000, 1000), chunks=(100, 100), dtype='float32')
baz[:] = 4.2
print(root.info)
```

```python exec="true" session="groups" source="above" result="ansi"
print(foo.info)
```

```python exec="true" session="groups" source="above" result="ansi"
print(bar.info_complete())
```

```python exec="true" session="groups" source="above" result="ansi"
print(baz.info)
```

Groups also have the [`zarr.Group.tree`][] method, e.g.:

```python exec="true" session="groups" source="above" result="ansi"
print(root.tree())
```


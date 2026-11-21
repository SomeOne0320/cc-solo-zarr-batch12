"""
Whole-group copy support.

This module implements the copy of an entire Zarr group into another group. The copy
preserves the Zarr format of the source: Zarr format 2 and format 3 groups keep their
respective storage layouts. Only the bytes that were actually written in the source are
copied, so unwritten chunks stay sparse in the destination.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

from zarr.core.array import AsyncArray
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import AsyncGroup, GroupMetadata
from zarr.core.metadata.io import save_metadata
from zarr.errors import CopyError
from zarr.storage._common import _contains_node_v2, _contains_node_v3, make_store_path
from zarr.storage._utils import _relativize_path

if TYPE_CHECKING:
    from zarr.core.common import ZarrFormat
    from zarr.storage import StoreLike, StorePath
    from zarr.types import AnyAsyncArray

IfExistsLiteral = Literal["raise", "skip"]

__all__ = [
    "copy",
    "copy_all",
    "copy_store",
]


async def _open_group(store_like: StoreLike, *, path: str | None, role: str) -> AsyncGroup:
    """
    Open an existing group from a store-like object, inferring its Zarr format when needed,
    and raising a clear error if the location does not contain a group.
    """
    store_path = await make_store_path(store_like, path=path)
    try:
        # ``zarr_format=None`` makes AsyncGroup.open infer the format from the metadata
        # documents present in storage (zarr.json for format 3, .zgroup for format 2).
        return await AsyncGroup.open(store_path, zarr_format=None)
    except FileNotFoundError as e:
        raise CopyError(
            f"The {role} does not contain a Zarr group at path {store_path.path!r}."
        ) from e


def _same_store(source_store: object, dest_store: object) -> bool:
    """
    Determine whether two store objects address the same underlying storage.

    Generic store equality is deliberately not used here: ``MemoryStore.__eq__`` compares the
    *contents* of two stores, so distinct in-memory stores that happen to hold the same data
    would wrongly be treated as the same store. Instead, identity is based on the backing
    dictionary for memory stores and on the resolved root path for local directory stores.
    """
    if source_store is dest_store:
        return True

    from zarr.storage._local import LocalStore
    from zarr.storage._memory import MemoryStore

    if isinstance(source_store, MemoryStore) and isinstance(dest_store, MemoryStore):
        # read-only views wrap the same underlying dictionary as their writable counterpart
        return source_store._store_dict is dest_store._store_dict
    if isinstance(source_store, LocalStore) and isinstance(dest_store, LocalStore):
        return source_store.root.resolve() == dest_store.root.resolve()
    return False


def _check_overlap(source: AsyncGroup, dest: AsyncGroup) -> None:
    """
    Reject copies where the source and destination group live in the same store and their
    storage locations overlap (i.e. one is a prefix of the other or they are identical).
    """
    if not _same_store(source.store_path.store, dest.store_path.store):
        return

    source_path = source.store_path.path
    dest_path = dest.store_path.path
    if (
        source_path == dest_path
        or source_path == ""
        or dest_path == ""
        or source_path.startswith(f"{dest_path}/")
        or dest_path.startswith(f"{source_path}/")
    ):
        raise CopyError(
            "The source and destination groups overlap in the same store "
            f"(source path {source_path!r}, destination path {dest_path!r}). "
            "Copying a group onto itself or into a nested location within itself is not supported."
        )


def _check_supported_stores(source: AsyncGroup, dest: AsyncGroup) -> None:
    """
    Only in-memory and local directory stores are supported, and only within the same store
    type and Zarr format.
    """
    from zarr.storage._local import LocalStore
    from zarr.storage._memory import MemoryStore

    supported = (MemoryStore, LocalStore)
    source_store = source.store_path.store
    dest_store = dest.store_path.store
    if not isinstance(source_store, supported) or not isinstance(dest_store, supported):
        raise TypeError(
            "Copying is only supported for in-memory (MemoryStore) and local directory "
            f"(LocalStore) stores. Got {type(source_store).__name__} and "
            f"{type(dest_store).__name__}."
        )
    if not isinstance(source_store, type(dest_store)):
        raise TypeError(
            "The source and destination stores must use the same store type. Got "
            f"{type(source_store).__name__} and {type(dest_store).__name__}."
        )

    source_format = source.metadata.zarr_format
    dest_format = dest.metadata.zarr_format
    if source_format != dest_format:
        raise ValueError(
            "The source and destination groups must use the same Zarr format. "
            f"Got Zarr format {source_format} for the source and {dest_format} for the destination."
        )


async def _node_exists(store_path: StorePath, zarr_format: ZarrFormat) -> bool:
    """Return whether an array or group already exists at ``store_path``."""
    if zarr_format == 3:
        node_kind = await _contains_node_v3(store_path)
    else:
        node_kind = await _contains_node_v2(store_path)
    return node_kind != "nothing"


def _strip_consolidated_metadata(metadata: GroupMetadata) -> GroupMetadata:
    """
    Consolidated metadata describes the layout of a group in its original store and must not
    be copied verbatim into another group. Drop it so the destination only carries regular
    metadata documents.
    """
    if metadata.consolidated_metadata is None:
        return metadata
    return dataclasses.replace(metadata, consolidated_metadata=None)


def _is_within(path: str, prefix: str) -> bool:
    """Return whether ``path`` is equal to ``prefix`` or nested below it."""
    return path == prefix or path.startswith(f"{prefix}/")


async def _copy_array_data(source_array: AnyAsyncArray, dest_array_path: StorePath) -> int:
    """
    Copy the raw, already-written chunk (or shard) objects of ``source_array`` into the array
    at ``dest_array_path``. Chunks that were never written have no storage object and are
    skipped, which keeps the destination sparse.

    Returns the number of bytes copied.
    """
    source_path = source_array.store_path.path
    source_store = source_array.store_path.store
    prototype = default_buffer_prototype()

    # The keys produced by ``_iter_shard_keys`` are relative to the array's store path; for
    # sharded arrays they are shard keys, otherwise they are chunk keys. Intersecting them
    # with the objects that actually exist in storage naturally excludes unwritten
    # chunks/shards.
    candidate_keys = set(source_array._iter_shard_keys())
    stored_relative_keys = [
        _relativize_path(path=key, prefix=source_path)
        async for key in source_store.list_prefix(prefix=source_path)
        if _relativize_path(path=key, prefix=source_path) in candidate_keys
    ]

    bytes_copied = 0
    for relative_key in stored_relative_keys:
        value = await (source_array.store_path / relative_key).get(prototype=prototype)
        if value is None:
            # the object disappeared between listing and reading; nothing to copy
            continue
        await (dest_array_path / relative_key).set(value)
        bytes_copied += len(value)
    return bytes_copied


async def _copy_group(
    source: AsyncGroup,
    dest: AsyncGroup,
    *,
    if_exists: IfExistsLiteral,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Shared implementation of the group copy. The destination group itself is assumed to exist
    already; only its contents (and its attributes) are copied.
    """
    if if_exists not in ("raise", "skip"):
        raise ValueError(
            f"Invalid value for 'if_exists': {if_exists!r}. Expected 'raise' or 'skip'."
        )

    # Resolve the full plan first so that a conflict (in the default 'raise' mode) aborts the
    # operation before anything is written to the destination.
    plan: list[tuple[str, AnyAsyncArray | AsyncGroup]] = [
        (path, node) async for path, node in source.members(max_depth=None)
    ]

    conflicts: set[str] = set()
    zarr_format = source.metadata.zarr_format
    for relative_path, _ in plan:
        if await _node_exists(dest.store_path / relative_path, zarr_format=zarr_format):
            conflicts.add(relative_path)

    def is_skipped(relative_path: str) -> bool:
        # A node is skipped when it conflicts itself or is nested inside a conflicting group:
        # skipping a group skips the complete contents of that group.
        return any(_is_within(relative_path, conflict) for conflict in conflicts)

    if conflicts and if_exists == "raise":
        formatted = ", ".join(repr(path) for path in sorted(conflicts))
        raise CopyError(
            "Refusing to copy because the following nodes already exist in the destination "
            f"group: {formatted}. Use if_exists='skip' to leave existing nodes untouched."
        )

    n_copied = sum(
        1
        for relative_path, node in plan
        if isinstance(node, AsyncArray) and not is_skipped(relative_path)
    )
    n_skipped = sum(
        1
        for relative_path, node in plan
        if isinstance(node, AsyncArray) and is_skipped(relative_path)
    )

    if dry_run:
        # Report what would happen without touching the destination.
        return n_copied, n_skipped, 0

    # Copy the root group's attributes. The destination group already exists, so its metadata
    # document is rewritten with the source attributes. This is the first write, and it only
    # happens after the conflict check above, so a failed preflight never leaves a partial
    # copy behind.
    await save_metadata(dest.store_path, _strip_consolidated_metadata(source.metadata))

    n_bytes_copied = 0
    for relative_path, node in plan:
        if is_skipped(relative_path):
            continue

        dest_node_path = dest.store_path / relative_path
        if isinstance(node, AsyncGroup):
            await save_metadata(
                dest_node_path, _strip_consolidated_metadata(node.metadata), ensure_parents=True
            )
        else:
            # persist the array metadata (shape, chunks, dtype, fill value, codecs, attrs, ...)
            await save_metadata(dest_node_path, node.metadata, ensure_parents=True)
            n_bytes_copied += await _copy_array_data(node, dest_node_path)

    return n_copied, n_skipped, n_bytes_copied


async def copy(
    source: AsyncGroup,
    dest: AsyncGroup,
    *,
    if_exists: IfExistsLiteral = "raise",
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Copy the contents of one Zarr group into another group.

    The source group may be located at the root of a store or in a nested directory; the
    destination must be an already existing group. After the copy, the destination contains
    the same sub-groups, arrays, attributes, shapes, chunk shapes, data types and fill values
    as the source. Zarr format 2 and format 3 groups keep their respective storage layouts,
    and only chunks that were written in the source are copied, so unwritten regions remain
    sparse in the destination.

    Only the contents of the source group are copied: sibling groups elsewhere in the source
    store are left untouched, and existing content of the destination outside of copied node
    names is preserved.

    Parameters
    ----------
    source : AsyncGroup
        The group to copy.
    dest : AsyncGroup
        An existing group to copy into.
    if_exists : {"raise", "skip"}, default "raise"
        What to do when a node with the same name as a source node already exists in the
        destination group.

        - ``"raise"`` (the default): raise a
          [`CopyError`][zarr.errors.CopyError] before writing anything.
        - ``"skip"``: leave the existing destination node, including all of its contents,
          untouched and copy every other branch normally. All arrays nested below a skipped
          group are counted as skipped.
    dry_run : bool, default False
        If True, inspect the source and destination and report conflicts and the work that
        would be performed, but do not write anything to the destination. The number of bytes
        copied is always zero for a dry run.

    Returns
    -------
    tuple of int
        A tuple of ``(n_copied, n_skipped, n_bytes_copied)``: the number of arrays that were
        copied, the number of arrays that were skipped because of a name conflict, and the
        total number of data bytes that were copied. In a dry run the third value is zero.

    Raises
    ------
    CopyError
        If a name conflict is encountered with ``if_exists="raise"``, or if the source and
        destination overlap within the same store.
    TypeError
        If the stores are not in-memory or local directory stores, or if the store types or
        Zarr formats do not match.
    ValueError
        If ``if_exists`` is not one of ``"raise"`` or ``"skip"``, or the Zarr formats of the
        source and destination groups differ.
    """
    _check_supported_stores(source, dest)
    _check_overlap(source, dest)
    return await _copy_group(source, dest, if_exists=if_exists, dry_run=dry_run)


async def copy_all(
    source: AsyncGroup,
    dest: AsyncGroup,
    *,
    if_exists: IfExistsLiteral = "raise",
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Copy the full contents of a group into another group.

    This is an alias for [`copy`][zarr.api.asynchronous.copy]: the whole hierarchy below the
    source group, including sub-groups, arrays, attributes and written chunk data, is copied
    into the destination group.

    Parameters
    ----------
    source : AsyncGroup
        The group to copy.
    dest : AsyncGroup
        An existing group to copy into.
    if_exists : {"raise", "skip"}, default "raise"
        See [`copy`][zarr.api.asynchronous.copy].
    dry_run : bool, default False
        See [`copy`][zarr.api.asynchronous.copy].

    Returns
    -------
    tuple of int
        See [`copy`][zarr.api.asynchronous.copy].
    """
    return await copy(source, dest, if_exists=if_exists, dry_run=dry_run)


async def copy_store(
    source: StoreLike,
    dest: StoreLike,
    *,
    source_path: str | None = None,
    dest_path: str | None = None,
    if_exists: IfExistsLiteral = "raise",
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Copy a group from one store into a group in another store (or another location in a
    store).

    This is a store-oriented wrapper around [`copy`][zarr.api.asynchronous.copy]: the source
    group is opened from ``source`` at ``source_path`` and the destination group is opened
    from ``dest`` at ``dest_path``. The destination group must already exist. Only in-memory
    and local directory stores of the same type and Zarr format are supported.

    Parameters
    ----------
    source : StoreLike
        The store-like object containing the source group. See the
        [storage documentation in the user guide][user-guide-store-like] for valid values.
    dest : StoreLike
        The store-like object containing the already existing destination group.
    source_path : str or None, optional
        The path to the source group within ``source``. If None, the store root is used.
    dest_path : str or None, optional
        The path to the destination group within ``dest``. If None, the store root is used.
    if_exists : {"raise", "skip"}, default "raise"
        See [`copy`][zarr.api.asynchronous.copy].
    dry_run : bool, default False
        See [`copy`][zarr.api.asynchronous.copy].

    Returns
    -------
    tuple of int
        See [`copy`][zarr.api.asynchronous.copy].
    """
    source_group = await _open_group(source, path=source_path, role="source store")
    dest_group = await _open_group(dest, path=dest_path, role="destination store")
    # validate store/format compatibility before doing any work, so unsupported combinations
    # fail with a clear error rather than deep inside traversal or write code paths.
    _check_supported_stores(source_group, dest_group)
    return await copy(source_group, dest_group, if_exists=if_exists, dry_run=dry_run)

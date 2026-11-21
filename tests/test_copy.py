from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import zarr
import zarr.api.asynchronous
from zarr.errors import CopyError
from zarr.storage import LocalStore, MemoryStore

if TYPE_CHECKING:
    from pathlib import Path

    from zarr.core.common import ZarrFormat


def _build_source_group(store: MemoryStore | LocalStore, zarr_format: ZarrFormat) -> zarr.Group:
    """
    Build a group exercising all node kinds: root attributes, an empty group, a nested group
    with attributes, a partially written (sparse) array and a fully sparse array.
    """
    root = zarr.open_group(store, mode="w", zarr_format=zarr_format)
    root.attrs["root_attr"] = "root-value"

    root.create_group("empty_group", attributes={"empty": [True]})

    sub = root.create_group("sub", attributes={"sub_attr": {"k": 1}})
    sparse = sub.create_array(
        "sparse",
        shape=(6, 4),
        chunks=(2, 4),
        dtype="int32",
        fill_value=-1,
        attributes={"units": "metres"},
    )
    data = np.arange(24, dtype="int32").reshape(6, 4)
    # only two of the three chunks are written
    sparse[:4, :] = data[:4, :]

    root.create_array("all_empty", shape=(3,), chunks=(3,), dtype="float64", fill_value=7.5)
    return root


def _assert_group_roundtrip(
    source: zarr.Group,
    dest_store: MemoryStore | LocalStore,
    zarr_format: ZarrFormat,
) -> None:
    """Reopen the destination and assert everything from the source is preserved."""
    dest = zarr.open_group(dest_store, mode="r")

    # sub-groups, including the empty one
    assert set(dest.keys()) == {"empty_group", "sub", "all_empty"}
    assert isinstance(dest["empty_group"], zarr.Group)
    assert dest["empty_group"].attrs["empty"] == [True]
    assert isinstance(dest["sub"], zarr.Group)
    assert dest["sub"].attrs["sub_attr"] == {"k": 1}

    # root attributes
    assert dest.attrs["root_attr"] == "root-value"

    # sparse array: shape, chunks, dtype, fill value, attributes and data
    source_sparse = source["sub/sparse"]
    copied_sparse = dest["sub/sparse"]
    assert copied_sparse.shape == source_sparse.shape == (6, 4)
    assert copied_sparse.chunks == source_sparse.chunks == (2, 4)
    assert copied_sparse.dtype == np.dtype("int32")
    assert copied_sparse.fill_value == -1
    assert copied_sparse.attrs["units"] == "metres"
    assert_array_equal(copied_sparse[:], source_sparse[:])
    # sparsity is preserved
    assert copied_sparse.nchunks_initialized == source_sparse.nchunks_initialized == 2

    # fully sparse array keeps its fill value and has no written chunks
    copied_empty = dest["all_empty"]
    assert copied_empty.shape == (3,)
    assert copied_empty.chunks == (3,)
    assert copied_empty.dtype == np.dtype("float64")
    assert copied_empty.fill_value == 7.5
    assert copied_empty.nchunks_initialized == 0
    assert_array_equal(copied_empty[:], np.full((3,), 7.5))

    # the exact storage layout (set of keys / files) is retained
    if isinstance(dest_store, MemoryStore) and isinstance(source.store, MemoryStore):
        assert set(dest_store._store_dict) == set(source.store._store_dict)
    else:
        source_root = source.store.root
        dest_root = dest_store.root
        source_files = sorted(
            p.relative_to(source_root) for p in source_root.rglob("*") if p.is_file()
        )
        dest_files = sorted(p.relative_to(dest_root) for p in dest_root.rglob("*") if p.is_file())
        assert source_files == dest_files


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_memory_store(zarr_format: ZarrFormat) -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    source = _build_source_group(source_store, zarr_format)
    zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)

    source_group = zarr.open_group(source_store, mode="r")
    dest_group = zarr.open_group(dest_store, mode="a")
    n_copied, n_skipped, n_bytes = await zarr.api.asynchronous.copy(
        source_group._async_group, dest_group._async_group
    )

    assert n_copied == 2
    assert n_skipped == 0
    assert n_bytes > 0
    # the byte count matches the raw sizes of the written chunk objects
    expected_bytes = sum(
        len(value)
        for key, value in source_store._store_dict.items()
        if "/c/" in key or (zarr_format == 2 and key.rsplit("/", 1)[-1][0].isdigit())
    )
    assert n_bytes == expected_bytes

    _assert_group_roundtrip(source, dest_store, zarr_format)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_copy_sync_local_store(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    source_root = tmp_path / "source.zarr"
    dest_root = tmp_path / "dest.zarr"
    source_store = LocalStore(source_root)
    dest_store = LocalStore(dest_root)
    source = _build_source_group(source_store, zarr_format)
    zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)

    n_copied, n_skipped, n_bytes = zarr.copy_store(source_store, dest_store)
    assert (n_copied, n_skipped) == (2, 0)
    assert n_bytes > 0

    _assert_group_roundtrip(source, dest_store, zarr_format)


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_store_paths(tmp_path: Path, zarr_format: ZarrFormat) -> None:
    # source group lives in a nested directory of its store
    source_store = LocalStore(tmp_path / "store_a")
    dest_store = LocalStore(tmp_path / "store_b")
    source = _build_source_group(source_store, zarr_format)
    zarr.open_group(dest_store, mode="w", path="nested/dest", zarr_format=zarr_format)
    # an unrelated group elsewhere in the destination store must not be affected
    neighbour = zarr.open_group(dest_store, mode="a", path="neighbour", zarr_format=zarr_format)
    neighbour.create_array("x", chunks=(2,), data=np.ones(2, dtype="int8"))

    stats = await zarr.api.asynchronous.copy_store(
        source_store,
        dest_store,
        source_path="",
        dest_path="nested/dest",
    )
    assert stats[:2] == (2, 0)
    assert stats[2] > 0

    copied = zarr.open_group(dest_store, mode="r", path="nested/dest")
    assert set(copied.keys()) == {"empty_group", "sub", "all_empty"}
    assert_array_equal(copied["sub/sparse"][:], source["sub/sparse"][:])

    # the neighbour group next to the copied group is untouched
    neighbour_after = zarr.open_group(dest_store, mode="r", path="neighbour")
    assert_array_equal(neighbour_after["x"][:], np.ones(2, dtype="int8"))
    assert "neighbour" not in copied


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_conflict_raises_before_any_write(zarr_format: ZarrFormat) -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    _build_source_group(source_store, zarr_format)
    dest_group = zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    # pre-existing array at one of the destination names
    dest_group.create_array("all_empty", shape=(1,), chunks=(1,), dtype="float64")

    before = dict(dest_store._store_dict)
    with pytest.raises(CopyError, match="all_empty"):
        await zarr.api.asynchronous.copy_store(source_store, dest_store)
    # nothing was written before the conflict was detected
    assert dest_store._store_dict == before


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_conflict_dry_run_raises_without_writes(zarr_format: ZarrFormat) -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    _build_source_group(source_store, zarr_format)
    dest_group = zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    dest_group.create_group("sub")

    before = dict(dest_store._store_dict)
    with pytest.raises(CopyError):
        await zarr.api.asynchronous.copy_store(source_store, dest_store, dry_run=True)
    assert dest_store._store_dict == before


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_skip_array_conflict(zarr_format: ZarrFormat) -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    source = _build_source_group(source_store, zarr_format)
    dest_group = zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    existing = dest_group.create_array("all_empty", shape=(9,), chunks=(3,), dtype="float64")
    existing[:] = np.full((9,), 3.25)

    n_copied, n_skipped, n_bytes = await zarr.api.asynchronous.copy_store(
        source_store, dest_store, if_exists="skip"
    )
    # sparse is copied, all_empty is skipped (whole node, including its data)
    assert n_copied == 1
    assert n_skipped == 1
    assert n_bytes > 0

    result = zarr.open_group(dest_store, mode="r")
    assert result["all_empty"].shape == (9,)
    assert_array_equal(result["all_empty"][:], np.full((9,), 3.25))
    assert_array_equal(result["sub/sparse"][:], source["sub/sparse"][:])


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_skip_group_conflict_skips_subtree(zarr_format: ZarrFormat) -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    _build_source_group(source_store, zarr_format)
    dest_group = zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    dest_group.create_group("sub")

    n_copied, n_skipped, n_bytes = await zarr.api.asynchronous.copy_store(
        source_store, dest_store, if_exists="skip"
    )
    # all_empty copied (with zero written chunks); the array nested below the conflicting
    # 'sub' group is skipped together with the whole subtree
    assert n_copied == 1
    assert n_skipped == 1
    assert n_bytes == 0

    result = zarr.open_group(dest_store, mode="r")
    assert "sparse" not in result["sub"]
    assert_array_equal(result["all_empty"][:], np.full((3,), 7.5))
    # other branch still copied
    assert result["empty_group"].attrs["empty"] == [True]


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_dry_run_reports_and_does_not_write(zarr_format: ZarrFormat) -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    _build_source_group(source_store, zarr_format)
    zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    group_metadata_keys = set(dest_store._store_dict)

    stats = await zarr.api.asynchronous.copy_store(source_store, dest_store, dry_run=True)
    assert stats == (2, 0, 0)
    # the destination still only contains its own group metadata
    assert set(dest_store._store_dict) == group_metadata_keys

    # dry run with a conflict + skip reports the skip, still without writing
    dest_with_conflict = MemoryStore()
    dg = zarr.open_group(dest_with_conflict, mode="w", zarr_format=zarr_format)
    dg.create_array("all_empty", shape=(1,), chunks=(1,), dtype="float64")
    skip_stats = await zarr.api.asynchronous.copy_store(
        source_store, dest_with_conflict, dry_run=True, if_exists="skip"
    )
    assert skip_stats == (1, 1, 0)
    assert "sub" not in dg


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_overlapping_locations_rejected(zarr_format: ZarrFormat) -> None:
    store = MemoryStore()
    outer = zarr.open_group(store, mode="w", path="outer", zarr_format=zarr_format)
    inner = zarr.open_group(store, mode="a", path="outer/inner", zarr_format=zarr_format)

    with pytest.raises(CopyError, match="overlap"):
        await zarr.api.asynchronous.copy(outer._async_group, inner._async_group)
    with pytest.raises(CopyError, match="overlap"):
        await zarr.api.asynchronous.copy(outer._async_group, outer._async_group)

    # disjoint locations within the same store are allowed
    other = zarr.open_group(store, mode="a", path="other", zarr_format=zarr_format)
    outer.create_array("a", chunks=(2,), data=np.ones(2, dtype="int8"))
    stats = await zarr.api.asynchronous.copy(outer._async_group, other._async_group)
    assert stats[0] == 1
    assert_array_equal(
        zarr.open_group(store, path="other", mode="r")["a"][:], np.ones(2, dtype="int8")
    )


async def test_copy_format_mismatch_rejected() -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    _build_source_group(source_store, 2)
    zarr.open_group(dest_store, mode="w", zarr_format=3)

    with pytest.raises(ValueError, match="same Zarr format"):
        await zarr.api.asynchronous.copy_store(source_store, dest_store)


async def test_copy_store_type_mismatch_rejected(tmp_path: Path) -> None:
    source_store = MemoryStore()
    dest_store = LocalStore(tmp_path / "dest")
    _build_source_group(source_store, 3)
    zarr.open_group(dest_store, mode="w", zarr_format=3)

    with pytest.raises(TypeError, match="same store type"):
        await zarr.api.asynchronous.copy_store(source_store, dest_store)


async def test_copy_unsupported_store_rejected(tmp_path: Path) -> None:
    # ZipStore is neither a MemoryStore nor a LocalStore and is not supported for copying
    source_store = MemoryStore()
    _build_source_group(source_store, 3)
    dest_zip = await zarr.storage.ZipStore.open(path=tmp_path / "dest.zip", mode="w")
    try:
        # the destination must be an existing group; create it before attempting the copy
        await zarr.api.asynchronous.open_group(store=dest_zip, mode="a", zarr_format=3)
        with pytest.raises(TypeError, match="MemoryStore"):
            await zarr.api.asynchronous.copy_store(source_store, dest_zip)
    finally:
        dest_zip.close()


async def test_copy_invalid_if_exists() -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    _build_source_group(source_store, 3)
    zarr.open_group(dest_store, mode="w", zarr_format=3)
    with pytest.raises(ValueError, match="if_exists"):
        await zarr.api.asynchronous.copy_store(source_store, dest_store, if_exists="replace")  # type: ignore[arg-type]


async def test_copy_missing_group_raises() -> None:
    with pytest.raises(CopyError, match="source store"):
        await zarr.api.asynchronous.copy_store(MemoryStore(), MemoryStore())


async def test_copy_all_alias() -> None:
    source_store = MemoryStore()
    dest = MemoryStore()
    _build_source_group(source_store, 3)
    zarr.open_group(dest, mode="w", zarr_format=3)
    stats = await zarr.api.asynchronous.copy_all(
        zarr.open_group(source_store, mode="r")._async_group,
        zarr.open_group(dest, mode="a")._async_group,
    )
    assert stats[0] == 2
    assert stats[1] == 0
    assert stats[2] > 0


def test_sync_and_async_entry_points_match() -> None:
    zarr_format: Literal[2, 3] = 3
    source_store = MemoryStore()
    dest_store = MemoryStore()
    source = _build_source_group(source_store, zarr_format)
    zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)

    sync_stats = zarr.copy_store(source_store, dest_store)
    assert isinstance(sync_stats, tuple)
    assert len(sync_stats) == 3

    fresh_dest = MemoryStore()
    zarr.open_group(fresh_dest, mode="w", zarr_format=zarr_format)
    async_stats = zarr.core.sync.sync(zarr.api.asynchronous.copy_store(source_store, fresh_dest))
    assert sync_stats == async_stats

    # group-object based entry points agree as well
    group_dest = MemoryStore()
    gd = zarr.open_group(group_dest, mode="w", zarr_format=zarr_format)
    group_stats = zarr.copy(
        zarr.open_group(source_store, mode="r"),
        gd,
    )
    assert group_stats == sync_stats
    _assert_group_roundtrip(source, group_dest, zarr_format)


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_preserves_sharded_layout(zarr_format: ZarrFormat) -> None:
    if zarr_format == 2:
        pytest.skip("sharding is a Zarr format 3 feature")
    source_store = MemoryStore()
    dest_store = MemoryStore()
    source = zarr.open_group(source_store, mode="w", zarr_format=zarr_format)
    sharded = source.create_array(
        "sharded",
        shape=(8, 8),
        chunks=(2, 2),
        shards=(4, 4),
        dtype="int16",
        fill_value=-3,
    )
    sharded[0:2, 0:2] = np.ones((2, 2), dtype="int16")
    sharded[6:8, 6:8] = np.full((2, 2), 9, dtype="int16")

    zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    stats = await zarr.api.asynchronous.copy_store(source_store, dest_store)
    assert stats[0] == 1
    assert stats[1] == 0
    assert stats[2] > 0

    # raw storage layout is identical, including the shard object keys
    assert set(dest_store._store_dict) == set(source_store._store_dict)
    copied = zarr.open_group(dest_store, mode="r")["sharded"]
    assert copied.shards == (4, 4)
    assert copied.chunks == (2, 2)
    assert copied.nchunks_initialized == sharded.nchunks_initialized
    assert_array_equal(copied[:], sharded[:])


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_nested_group_with_similar_sibling_names(zarr_format: ZarrFormat) -> None:
    # groups 'a' and 'ab' share a name prefix; listing 'a' must not pick up 'ab'
    source_store = MemoryStore()
    root = zarr.open_group(source_store, mode="w", zarr_format=zarr_format)
    first = root.create_array("a/data", shape=(4,), chunks=(2,), dtype="int32")
    first[:] = np.arange(4, dtype="int32")
    second = root.create_array("ab/data", shape=(4,), chunks=(2,), dtype="int32")
    second[:] = np.arange(4, 8, dtype="int32")

    dest_store = MemoryStore()
    zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    nested_source = zarr.open_group(source_store, mode="r", path="a")
    stats = await zarr.api.asynchronous.copy(
        nested_source._async_group,
        zarr.open_group(dest_store, mode="a")._async_group,
    )
    assert stats[0] == 1
    assert not any(key.startswith("ab") for key in dest_store._store_dict)
    result = zarr.open_group(dest_store, mode="r")
    assert_array_equal(result["data"][:], np.arange(4, dtype="int32"))


@pytest.mark.parametrize("zarr_format", [2, 3])
async def test_copy_destination_extras_are_preserved(zarr_format: ZarrFormat) -> None:
    source_store = MemoryStore()
    dest_store = MemoryStore()
    _build_source_group(source_store, zarr_format)
    dest_group = zarr.open_group(dest_store, mode="w", zarr_format=zarr_format)
    # content outside the copied names must survive the copy
    extra = dest_group.create_array("extra", chunks=(2,), data=np.array([4, 5], dtype="int8"))

    await zarr.api.asynchronous.copy_store(source_store, dest_store)
    result = zarr.open_group(dest_store, mode="r")
    assert_array_equal(result["extra"][:], extra[:])
    assert set(result.keys()) == {"empty_group", "sub", "all_empty", "extra"}


async def test_copy_consolidated_metadata_is_not_propagated() -> None:
    import json

    source_store = MemoryStore()
    source = zarr.open_group(source_store, mode="w", zarr_format=3)
    source.create_group("sub").create_array("a", chunks=(2,), data=np.arange(4, dtype="int8"))
    # consolidation emits a warning that the feature is outside the Zarr v3 specification
    with pytest.warns(zarr.errors.ZarrUserWarning):
        await zarr.api.asynchronous.consolidate_metadata(source_store)
    assert "consolidated_metadata" in json.loads(source_store._store_dict["zarr.json"].to_bytes())

    dest_store = MemoryStore()
    zarr.open_group(dest_store, mode="w", zarr_format=3)
    await zarr.api.asynchronous.copy_store(source_store, dest_store)

    dest_root = json.loads(dest_store._store_dict["zarr.json"].to_bytes())
    assert "consolidated_metadata" not in dest_root
    assert_array_equal(
        zarr.open_group(dest_store, mode="r")["sub/a"][:], np.arange(4, dtype="int8")
    )

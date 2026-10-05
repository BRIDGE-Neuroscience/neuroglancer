"""Generates the zarr-vectors reader fixtures in this directory.

Run with a Python that has zarr>=3 and zarr-vectors 0.9.2 (the released format,
https://github.com/AllenInstitute/zarr-vectors-py tag v0.9.2):

    python testdata/datasource/zarr-vectors/generate.py

Every store holds the same handful of objects so the reader tests can check
that each codec / layout decodes to identical bytes.  `expected.json` records
the values the tests compare against.
"""

import json
import os
import shutil
import subprocess
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import zarr_vectors.core.arrays as zv_arrays  # noqa: E402
from zarr_vectors.constants import FORMAT_VERSION  # noqa: E402
from zarr_vectors.multiresolution.coarsen import build_pyramid  # noqa: E402
from zarr_vectors.types.graphs import write_graph  # noqa: E402
from zarr_vectors.types.meshes import write_mesh  # noqa: E402
from zarr_vectors.types.points import write_points  # noqa: E402
from zarr_vectors.types.polylines import write_polylines  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# zarr-vectors-tools' CLI, for the mesh pyramid (`mesh_lod`); that fixture is
# left as it is when the CLI is not available.
ZVTOOLS = os.environ.get("ZVTOOLS") or shutil.which("zvtools")


def icosphere(subdivisions):
    """Unit icosphere: (vertices, triangles)."""
    t = (1 + 5**0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
         [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
         [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    verts = [np.array(x, float) / np.linalg.norm(x) for x in v]
    for _ in range(subdivisions):
        cache = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                cache[key] = len(verts)
                verts.append(m / np.linalg.norm(m))
            return cache[key]

        f = [tri for a, b, c in f for tri in (
            [a, mid(a, b), mid(c, a)], [b, mid(b, c), mid(a, b)],
            [c, mid(c, a), mid(b, c)], [mid(a, b), mid(b, c), mid(c, a)])]
    return np.array(verts), np.array(f)


def path(name):
    return os.path.join(HERE, name + ".zarrvectors")


def fresh(name):
    p = path(name)
    if os.path.exists(p):
        shutil.rmtree(p)
    return p


def main():
    # Small row buckets keep the fixtures small; the reader must not assume
    # the writer's 65,536 / 16,384 defaults anyway.
    zv_arrays.OBJECT_ATTRIBUTE_ROW_BUCKET = 8
    zv_arrays.OBJECT_INDEX_MANIFEST_BUCKET = 8
    rng = np.random.default_rng(0)
    polys = [
        (np.cumsum(rng.normal(size=(20, 3)), axis=0) + 24).astype("float32")
        for _ in range(6)
    ]
    fa = [rng.random(len(p)).astype("float32") for p in polys]
    object_attributes = {
        "length": np.arange(6, dtype="float32") * 1.5,
        "kind": np.array([0, 1, 0, 1, 2, 2], dtype="int32"),
    }
    groups = {0: [0, 1, 2], 1: [3, 4, 5]}
    chunk_shape = (16.0, 16.0, 16.0)

    variants = [
        ("poly_raw", None, None),
        ("poly_zstd", "zstd", None),
        ("poly_blosc", "blosc", None),
        ("poly_gzip", [{"name": "gzip", "configuration": {"level": 5}}], None),
        ("poly_raw_shard", None, 2),
        ("poly_zstd_shard", "zstd", 2),
    ]
    for name, compressor, shard in variants:
        write_polylines(
            fresh(name),
            polys,
            chunk_shape=chunk_shape,
            vertex_attributes={"fa": fa},
            object_attributes=object_attributes,
            groups=groups,
            compressor=compressor,
            shard_shape=shard,
        )

    # Object-level arrays split over several chunks, as any store with more
    # than 65,536 objects (or 16,384 manifests) has.
    saved = (
        zv_arrays.OBJECT_ATTRIBUTE_ROW_BUCKET,
        zv_arrays.OBJECT_INDEX_MANIFEST_BUCKET,
    )
    zv_arrays.OBJECT_ATTRIBUTE_ROW_BUCKET = 4
    zv_arrays.OBJECT_INDEX_MANIFEST_BUCKET = 4
    try:
        write_polylines(
            fresh("poly_multichunk"),
            polys,
            chunk_shape=chunk_shape,
            object_attributes=object_attributes,
        )
    finally:
        (
            zv_arrays.OBJECT_ATTRIBUTE_ROW_BUCKET,
            zv_arrays.OBJECT_INDEX_MANIFEST_BUCKET,
        ) = saved

    write_polylines(
        fresh("poly_f64"),
        [p.astype("float64") for p in polys],
        chunk_shape=chunk_shape,
        dtype="float64",
    )
    write_polylines(
        fresh("poly_mc"),
        polys,
        chunk_shape=chunk_shape,
        vertex_attributes={
            "rgb": [rng.random((len(p), 3)).astype("float32") for p in polys]
        },
    )

    # A pyramid built over a compressed level 0: coarse levels come out raw.
    write_polylines(
        fresh("pyr_zstd"),
        polys,
        chunk_shape=chunk_shape,
        compressor="zstd",
        object_attributes={"length": object_attributes["length"]},
    )
    build_pyramid(path("pyr_zstd"), factors=[(2, 2)])

    points = (rng.random((60, 3)) * 30).astype("float32")
    point_ids = np.repeat(np.array([7, 42, 1000], dtype="int64"), 20)
    write_points(
        fresh("pts_raw"),
        points,
        chunk_shape=chunk_shape,
        vertex_attributes={"i": np.arange(60, dtype="float32")},
    )
    write_points(
        fresh("pts_sparse_ids"),
        points,
        chunk_shape=chunk_shape,
        object_ids=point_ids,
    )

    grid = np.array(
        [[x, y, 0] for x in range(4) for y in range(4)], dtype="float32"
    ) * 10
    faces = []
    for i in range(3):
        for j in range(3):
            a = i * 4 + j
            faces += [[a, a + 1, a + 4], [a + 1, a + 5, a + 4]]
    write_mesh(
        fresh("mesh_raw"),
        grid,
        np.array(faces),
        chunk_shape=(16.0, 16.0, 16.0),
        encoding="raw",
    )

    # Two branching trees.  Tree 0 leaves chunk (0,0,0) and comes back, so
    # vertex 4 sits mid-fragment there with its parent in chunk (1,0,0): the
    # cross-chunk link must replace the parent the row order implies.  Tree 1
    # is not the first root, so the writer leaves it out of depth-first order.
    skeleton_positions = np.array(
        [
            [4, 4, 4], [12, 4, 4], [20, 4, 4], [20, 12, 4],
            [12, 12, 4], [4, 12, 4], [12, 8, 8], [24, 4, 10],
            [4, 4, 20], [4, 4, 28], [4, 12, 28], [4, 20, 28], [12, 12, 20],
        ],
        dtype="float32",
    )
    # [child, parent]
    skeleton_edges = np.array(
        [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 1], [7, 2],
         [9, 8], [10, 9], [11, 10], [12, 9]],
        dtype="int64",
    )
    skeleton_object_ids = np.array([0] * 8 + [1] * 5, dtype="int64")
    write_graph(
        fresh("skel_raw"),
        skeleton_positions,
        skeleton_edges,
        chunk_shape=chunk_shape,
        kind="skeleton",
        object_ids=skeleton_object_ids,
    )

    # Two spheres, then a zarr-vectors-tools pyramid of decimated surfaces whose
    # chunks double per level: a multi-resolution mesh.
    sphere_v, sphere_f = icosphere(3)
    lod_centers, lod_radii = [(20.0, 20.0, 20.0), (44.0, 26.0, 22.0)], [12.0, 9.0]
    lod_positions = np.concatenate(
        [sphere_v * r + c for c, r in zip(lod_centers, lod_radii)]
    ).astype("float32")
    lod_faces = np.concatenate([sphere_f, sphere_f + len(sphere_v)])
    lod_object_ids = np.repeat(np.arange(2, dtype="int64"), len(sphere_v))
    if ZVTOOLS:
        write_mesh(
            fresh("mesh_lod"),
            lod_positions,
            lod_faces,
            chunk_shape=(16.0, 16.0, 16.0),
            object_ids=lod_object_ids,
            bounds=([0, 0, 0], [64, 48, 40]),
            encoding="raw",
        )
        subprocess.run(
            [ZVTOOLS, "pyramid", path("mesh_lod"), "--coarsen", "4,4",
             "--chunk-scale", "2,2", "--method", "mesh_decimate"],
            check=True,
        )
        # zarr-vectors-py's write_mesh stamps level 0 fragment_link_groups
        # and the pyramid stamps the levels it writes, so every level keeps
        # one face group per object.

    # One set of streamlines as two pyramids: replacement, and additive
    # (each object stored once, at the coarsest level that keeps it). The
    # union of an additive level's chain must equal the replacement level.
    if ZVTOOLS:
        walks = [
            (np.cumsum(rng.normal(scale=2.5, size=(16 + 2 * i, 3)), axis=0)
             + rng.random(3) * 40 + 12).astype("float32")
            for i in range(24)
        ]
        walks = [np.clip(w, 0.5, 63.5) for w in walks]
        for name, refinement in (("add_replace", []), ("add_additive", ["--refinement", "add"])):
            write_polylines(
                fresh(name),
                walks,
                chunk_shape=(32.0, 32.0, 32.0),
                bounds=([0, 0, 0], [64, 64, 64]),
                object_attributes={
                    "length": np.array([len(w) for w in walks], dtype="float32"),
                },
            )
            subprocess.run(
                [ZVTOOLS, "pyramid", path(name), "--coarsen", "1,1",
                 "--sparsity", "2,2", "--sparsity-strategy", "length",
                 "--method", "polyline", *refinement],
                check=True,
            )
        # A mesh pyramid stored additively: the viewer draws it as one level
        # of detail holding each object's parts from every level.
        write_mesh(
            fresh("mesh_add"),
            lod_positions,
            lod_faces,
            chunk_shape=(16.0, 16.0, 16.0),
            object_ids=lod_object_ids,
            bounds=([0, 0, 0], [64, 48, 40]),
            encoding="raw",
        )
        subprocess.run(
            [ZVTOOLS, "pyramid", path("mesh_add"), "--coarsen", "1",
             "--sparsity", "2", "--sparsity-strategy", "random",
             "--chunk-scale", "2", "--method", "mesh", "--refinement", "add"],
            check=True,
        )

    # Objects whose ids are small integers, then a zarr-vectors-tools
    # polyline pyramid: it writes dense rows into fragment_attributes/
    # segment_id, which collide with the ids. The viewer must take objects
    # from the manifests, never guess per value.
    tracks = [
        (np.cumsum(rng.normal(scale=3.0, size=(30, 3)), axis=0)
         + rng.uniform(4, 60, 3)).astype("float32")
        for _ in range(12)
    ]
    tracks = [np.clip(t, 0.5, 63.5) for t in tracks]
    if ZVTOOLS:
        write_polylines(
            fresh("ids_collide"),
            tracks,
            chunk_shape=(32.0, 32.0, 32.0),
            bounds=([0, 0, 0], [64, 64, 64]),
            object_attributes={
                "segment_id": np.arange(1, 13, dtype="uint64"),
                "length": np.array([len(t) for t in tracks], dtype="float32"),
            },
        )
        subprocess.run(
            [ZVTOOLS, "pyramid", path("ids_collide"), "--coarsen", "1",
             "--sparsity", "2", "--sparsity-strategy", "length",
             "--method", "polyline"],
            check=True,
        )
    # A store whose ids are its rows, with an int32 per-fragment id column.
    write_polylines(
        fresh("ids_int32"),
        tracks,
        chunk_shape=(32.0, 32.0, 32.0),
        bounds=([0, 0, 0], [64, 64, 64]),
    )
    from zarr_vectors.building import (  # noqa: E402
        create_fragment_attribute_array,
        get_resolution_level,
        open_store,
        read_all_object_manifests,
        read_vertex_fragment_index,
        write_chunk_fragment_attributes,
    )

    level = get_resolution_level(open_store(path("ids_int32"), mode="r+"), 0)
    owners = {}
    for row, manifest in enumerate(read_all_object_manifests(level)):
        for chunk, fragment in manifest:
            owners.setdefault(tuple(int(c) for c in chunk), {})[int(fragment)] = row
    create_fragment_attribute_array(level, "segment_id", dtype="int32")
    for chunk, rows in owners.items():
        column = np.zeros(len(read_vertex_fragment_index(level, chunk)), "int32")
        for fragment, row in rows.items():
            column[fragment] = row
        write_chunk_fragment_attributes(
            level, "segment_id", chunk, column, dtype=np.int32
        )
    # zarr-vectors-py's own pyramid writes no vertex attributes at its
    # coarse level: they are unknown there, not zero.
    write_polylines(
        fresh("attrs_coarse"),
        tracks,
        chunk_shape=(32.0, 32.0, 32.0),
        bounds=([0, 0, 0], [64, 64, 64]),
        vertex_attributes={
            "radius": [np.full(len(t), 2.0 + i, "float32") for i, t in enumerate(tracks)],
        },
    )
    build_pyramid(path("attrs_coarse"), factors=[(2, 2)])

    expected = {
        "mesh_lod": {
            "positions": lod_positions.tolist(),
            "faces": lod_faces.tolist(),
            "object_ids": lod_object_ids.tolist(),
        },
        "skeleton": {
            "positions": skeleton_positions.tolist(),
            "edges": skeleton_edges.tolist(),
            "object_ids": skeleton_object_ids.tolist(),
        },
        "generator": {
            "format_version": FORMAT_VERSION,
        },
        "polylines": [p.tolist() for p in polys],
        "fa": [a.tolist() for a in fa],
        "object_attributes": {k: v.tolist() for k, v in object_attributes.items()},
        "groups": {str(k): v for k, v in groups.items()},
        "points": points.tolist(),
        "point_object_ids": [7, 42, 1000],
        "mesh_vertices": grid.tolist(),
        "mesh_faces": faces,
    }
    with open(os.path.join(HERE, "expected.json"), "w") as f:
        json.dump(expected, f)


if __name__ == "__main__":
    main()

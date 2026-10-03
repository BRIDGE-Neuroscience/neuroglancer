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
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import zarr_vectors.core.arrays as zv_arrays  # noqa: E402
from zarr_vectors.constants import FORMAT_VERSION  # noqa: E402
from zarr_vectors.multiresolution.coarsen import build_pyramid  # noqa: E402
from zarr_vectors.types.meshes import write_mesh  # noqa: E402
from zarr_vectors.types.points import write_points  # noqa: E402
from zarr_vectors.types.polylines import write_polylines  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


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

    expected = {
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

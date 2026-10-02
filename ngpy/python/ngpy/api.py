# @license
# Copyright 2026 The Neuroglancer Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry points the ngpy page calls in the Pyodide worker.

Every function takes and returns JSON TEXT (or bytes), so nothing crosses the
JS/Python boundary as a live proxy and there is nothing to leak or destroy.
Async functions become JS Promises; none of them reaches zarr's blocking
``sync()`` except :func:`export_sync`, which the worker only calls through
``callPromising`` under its one-at-a-time mutex (JSPI).
"""

from __future__ import annotations

import json
import sys
import types
import typing


def _install_pil_stub() -> None:
    """Let ``neuroglancer.chunks`` import without Pillow (not loaded by default)."""
    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        stub = types.ModuleType("PIL")

        class _Missing:
            def __getattr__(self, name):
                raise ImportError(
                    "Pillow is not loaded in ngpy; call "
                    "`await ngpy.api.load_packages(['pillow'])` first"
                )

        stub.Image = _Missing()  # type: ignore[attr-defined]
        stub.__ngpy_stub__ = True  # type: ignore[attr-defined]
        sys.modules["PIL"] = stub


def boot(info_json: str) -> str:
    """Called once after the payload is unpacked."""
    info = json.loads(info_json)
    _install_pil_stub()
    from . import bridge

    bridge.page_info.update(info)
    import neuroglancer  # noqa: F401  (vendored upstream subset + ngpy shims)
    import zarr_vectors

    return json.dumps(
        {
            "python": sys.version.split()[0],
            "zarr_vectors": getattr(zarr_vectors, "__version__", "?"),
            "numpy": __import__("numpy").__version__,
            "zarr": __import__("zarr").__version__,
        }
    )


async def load_packages(names: list[str]) -> None:
    """Load Pyodide-distribution packages (e.g. ``pillow``, ``scipy``)."""
    import pyodide_js  # type: ignore[import-not-found]

    await pyodide_js.loadPackage(list(names))
    if "pillow" in names and getattr(sys.modules.get("PIL"), "__ngpy_stub__", False):
        del sys.modules["PIL"]


# -- bridge ---------------------------------------------------------------------


def client_state(message: str) -> str:
    from .bridge import handle_client_state

    return handle_client_state(message)


def action(message: str) -> None:
    from .bridge import handle_action

    handle_action(message)


# -- filter ---------------------------------------------------------------------


async def store_info(request_json: str) -> str:
    from .filter import engine

    request = json.loads(request_json)
    return json.dumps(await engine().store_info(request["source"]))


async def filter_evaluate(request_json: str) -> str:
    from .filter import engine

    return json.dumps(await engine().evaluate(json.loads(request_json)))


async def filter_passing_ids(request_json: str) -> str:
    from .filter import engine

    return json.dumps(await engine().passing_ids(json.loads(request_json)))


async def label_info(request_json: str) -> str:
    """Label list for the label filter: names from segment properties if given,
    otherwise the distinct labels present in the volume."""
    import numpy as np

    from . import labels

    request = json.loads(request_json)
    out: dict[str, typing.Any] = {"labels": [], "volume": None}
    props = None
    if request.get("propertiesUrl"):
        try:
            props = await labels.read_segment_properties(request["propertiesUrl"])
        except Exception as e:  # noqa: BLE001
            out["propertiesError"] = str(e)
    present = None
    if request.get("volumeUrl"):
        from .filter import engine

        eng = engine()
        key = f"{request['volumeUrl']}#{request.get('scale')}"
        volume = eng._volumes.get(key)
        if volume is None:
            volume = await labels.open_label_volume(
                request["volumeUrl"], scale_index=request.get("scale")
            )
            eng._volumes[key] = volume
        out["volume"] = volume.description
        present = {int(v) for v in np.unique(volume.data) if int(v) != 0}
    if props is not None:
        for i, name, color in zip(props["ids"], props["names"], props["colors"]):
            if present is not None and int(i) not in present:
                continue
            out["labels"].append({"id": int(i), "name": name, "color": color})
    elif present is not None:
        out["labels"] = [{"id": v, "name": str(v), "color": None} for v in sorted(present)]
    return json.dumps(out)


# -- export ---------------------------------------------------------------------


async def export(job_json: str):
    """``(content_type, body_bytes, summary_json)`` for one export job."""
    from .tract_export.browser import export_async
    from .tract_export.job import JobSpecError, parse_job
    from .tract_export.run import ExportRunError

    try:
        job = parse_job(json.loads(job_json))
        body, content_type, summary = await export_async(job)
    except (JobSpecError, ExportRunError, ValueError) as e:
        return ("text/plain", str(e).encode("utf-8"), json.dumps({"error": str(e)}))
    return (content_type, body, json.dumps(summary))


def export_sync(job_json: str):
    """:func:`export` driven to completion synchronously (needs JSPI).

    Called from JS via ``callPromising`` -- the suspender that lets both the
    awaited fetches and zarr's synchronous writer (``.zvf``) run.
    """
    from pyodide.ffi import run_sync  # type: ignore[import-not-found]

    return run_sync(export(job_json))

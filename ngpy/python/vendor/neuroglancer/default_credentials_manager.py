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

"""ngpy replacement for upstream ``neuroglancer.default_credentials_manager``.

Upstream forwards credentials from the Python process to the viewer.  Under ngpy
the hosted viewer obtains its own credentials (it is an ordinary Neuroglancer
page), so there is nothing for Python to forward.
"""


def set_boss_token(token):  # noqa: ARG001
    raise NotImplementedError(
        "ngpy: credentials are obtained by the hosted Neuroglancer page itself; "
        "set_boss_token is not supported."
    )

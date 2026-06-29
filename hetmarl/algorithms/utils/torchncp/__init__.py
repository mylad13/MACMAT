# Copyright 2020-2021 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# from __future__ import absolute_import


from .cfc_cell import CfCCell
from .wired_cfc_cell import WiredCfCCell

__all__ = ["CfC", "CfCCell", "LTC", "LTCCell", "WiredCfCCell"]


# LTCCell, CfC and LTC pull in the external `ncps` package; MACMAT itself uses
# only the vendored CfCCell/WiredCfCCell (and the local wirings). Import the
# ncps-backed exports lazily (PEP 562) so a default MACMAT run needs no `ncps`
# installed, while `from torchncp import LTCCell` still works when it is.
def __getattr__(name):
    if name == "LTCCell":
        from ncps.torch.ltc_cell import LTCCell
        return LTCCell
    if name == "CfC":
        from .cfc import CfC
        return CfC
    if name == "LTC":
        from .ltc import LTC
        return LTC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
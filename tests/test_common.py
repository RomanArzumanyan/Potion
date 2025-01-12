# Copyright 2025 Roman Arzumanyan.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http: // www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import python_potion.common as common


def make_flags(
        input="",
        gpu_id=0,
        time=3.0,
        num_retries=3,
        model_name="densenet_onnx",
        model_version=12,
        classes=1,
        url="0.0.0.0:8001",
        dump="",
        verbose=False,) -> argparse.Namespace:

    args = [
        "-i", input,
        "-g", str(gpu_id),
        "-t", str(time),
        "-n", str(num_retries),
        "-m", model_name,
        "-x", str(model_version),
        "-c", str(classes),
        "-u", url
    ]

    if len(dump):
        args.append(["-d", dump])

    if verbose:
        args.append("-v")

    parser = common.get_parser()
    return parser.parse_args(args)

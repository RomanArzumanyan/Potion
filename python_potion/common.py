# Copyright 2024 Roman Arzumanyan.
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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "This utility decodes input video, runs inference on it and prints out resuts. \n"
        "It can also dumps video track. \n"
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        required=True,
        choices=range(0, 99),
        help="GPU id, check nvidia-smi",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Encoded video file (read from)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        required=True,
        help="processing time, s.",
    )
    parser.add_argument(
        "-d",
        "--dump",
        type=str,
        required=False,
        default="",
        help="dump video filename without extension",
    )
    parser.add_argument(
        "-n",
        "--num_retries",
        type=int,
        required=False,
        default=3,
        help="number of attepts to respawn video reader in case of failure",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output, turned off by default",
    )
    parser.add_argument(
        "-m", "--model-name", type=str, required=True, help="Name of model"
    )
    parser.add_argument(
        "-x",
        "--model-version",
        type=str,
        required=False,
        default="",
        help="Version of model. Default is to use latest version.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        required=False,
        default=1,
        help="Batch size. Default is 1.",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        required=False,
        default=1,
        help="Number of class results to report. Default is 1.",
    )
    return parser

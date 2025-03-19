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
from multiprocessing import Queue, Process
from queue import Empty


def find_shortest_path(graph: dict, start, goal) -> list:
    """
    Find shortest path in graph

    Args:
        graph (dict): graph
        start: start point
        goal : end point

    Returns:
        list: list where every item is a node. Empty if no path exists
    """
    explored = []
    queue = [[start]]

    if start == goal:
        return [start]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in explored:
            neighbours = graph[node]
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                if neighbour == goal:
                    return new_path
            explored.append(node)

    return []


def drain(q: Queue) -> int:
    """
    Drain queue. Will not return unless :arg:`q` is empty.

    Args:
        q (Queue): input queue.

    Returns:
        int: drained data size.
    """
    size = 0
    while True:
        try:
            chunk = q.get_nowait()
            if chunk is None:
                return size
            size += len(chunk)

        except Empty:
            continue

        except ValueError:
            break
    return size


def get_dec_bench_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "This benchmark measures pure decoding speed \n"
    )
    parser.add_argument(
        "-p",
        "--num_procs",
        type=int,
        required=True,
        choices=range(0, 99),
        default=0,
        help="number of decoding processes",
    )
    return parser


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
        choices=range(-1, 99),
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
        type=float,
        required=False,
        default=-1.0,
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
    parser.add_argument(
        "-buf_queue_size",
        "--buf_queue_size",
        type=int,
        required=False,
        default=1024,
        help="Buffer queue size in chunks. Default is 1024.",
    )
    return parser

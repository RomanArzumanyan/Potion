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


import subprocess
import json

from typing import Dict
from io import BytesIO
import python_vali as vali
from multiprocessing import Queue
import logging
from multiprocessing.synchronize import Event as SyncEvent

logger = logging.getLogger(__file__)


def get_stream_params(url: str) -> Dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        url,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout = proc.communicate()[0]

    bio = BytesIO(stdout)
    json_out = json.load(bio)

    params = {}
    if not "streams" in json_out:
        raise ValueError("No stream parameters found")

    for stream in json_out["streams"]:
        if stream["codec_type"] == "video":
            params["width"] = stream["width"]
            params["height"] = stream["height"]
            params["framerate"] = float(eval(stream["avg_frame_rate"]))

            codec_name = stream["codec_name"]
            is_h264 = True if codec_name == "h264" else False
            is_hevc = True if codec_name == "hevc" else False

            if not is_h264 and not is_hevc:
                raise ValueError(
                    f"Unsupported codec {codec_name}: neither h264 nor hevc"
                )

            params["codec"] = stream["codec_name"]
            pix_fmt = stream["pix_fmt"]
            is_yuv420 = pix_fmt == "yuv420p" or pix_fmt == "yuvj420p"

            if not is_yuv420:
                raise ValueError(
                    f"Unsupported format {pix_fmt}. Only yuv420 for now"
                )

            params["format"] = (
                vali.PixelFormat.NV12 if is_yuv420 else vali.PixelFormat.YUV444
            )

            return params

    raise ValueError("No video streams found")


def buf_stream(url: str, params: Dict, buf_queue: Queue, stop_event: SyncEvent) -> None:
    # Prepare FFMpeg arguments
    codec = params["codec"]
    bsf_name = codec + "_mp4toannexb,dump_extra=all"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "fatal",
        "-i",
        url,
        "-c:v",
        "copy",
        "-bsf:v",
        bsf_name,
        "-f",
        codec,
        "pipe:1",
    ]

    # Run ffmpeg in subprocess and redirect it's output to pipe
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    # Read from pipe and put into queue
    read_size = 4096
    while not stop_event.is_set():
        try:
            bytes = proc.stdout.read(read_size)
            if not len(bytes):
                break
            buf_queue.put(bytes)

        except ValueError:
            break

        except EOFError:
            logger.info(f"EOF: complete reading {url}.")

    proc.terminate()

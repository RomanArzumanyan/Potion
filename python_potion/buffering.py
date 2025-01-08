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
from enum import Enum
from multiprocessing.synchronize import Event as SyncEvent

LOGGER = logging.getLogger(__file__)


class FFMpegProcState(Enum):
    RUNNING = 0,
    EOF = 1,
    ERROR = 2


class StreamBuffer:
    def __init__(self, url: str, settings: Dict):
        self.num_retries = settings['num_retries']
        self.err_cnt = 0
        self.params = self.get_params_impl(url)
        self.url = url

    def get_params(self) -> Dict:
        """
        Video params getter.

        Returns:
            Dict: dictionary with video stream params.
        """
        return self.params

    def get_params_impl(self, url: str) -> Dict:
        """
        Get video stream parameters via ffprobe.
        If there are multiple video stream, 1st stream params will be returned.

        Args:
            url (str): video URL

        Raises:
            ValueError: with meaningful description if things go wrong.

        Returns:
            Dict: dictionary with video stream params.
        """

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

    def check_update_state(self) -> tuple[bool, FFMpegProcState]:
        """
        Checks FFMpeg process state and return tuple which describes it.

        Returns:
            tuple[bool, FFMpegProcState]: tuple[0] is True if FFMpeg is running,
            False otherwise. tuple[1] is FFMpeg process state.
        """

        p = self.proc.poll()

        if p is None:
            return True, FFMpegProcState.RUNNING
        else:
            if p == 0:
                return False, FFMpegProcState.EOF
            else:
                self.err_cnt += 1
                return False, FFMpegProcState.ERROR

    def format_by_codec(self) -> str:
        """
        Get output format by codec

        Returns:
            str: format name
        """

        codec = self.params["codec"]
        if codec == "h264" or codec == "hevc" or codec == "av1":
            return "mpegts"
        elif codec == "vp8" or codec == "vp9":
            return "webm"
        else:
            raise RuntimeError(f"Unsupported codec: {codec}")

    def run_ffmpeg(self) -> None:
        """
        Run FFMpeg in subprocess and redirect output to pipe.
        """

        # Prepare FFMpeg arguments
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "fatal",
            "-i",
            self.url,
            "-map",
            "v:0",
            "-c:v",
            "copy",
            "-c:a",
            "none",
            "-f",
            self.format_by_codec(),
            "pipe:1",
        ]

        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def ffmpeg_needs_respawn(self) -> bool:
        """
        Check if FFMpeg process needs to be respawned.

        Returns:
            bool: True if FFMpeg needs to be respawned, False otherwise.
        """

        proc_status = self.check_update_state()
        alive = proc_status[0]
        if alive:
            return False
        else:
            death_reason = proc_status[1]
            if death_reason == FFMpegProcState.EOF:
                LOGGER.info("FFMpeg buffering process exited normally.")
                return False
            else:
                if self.err_cnt < self.num_retries:
                    LOGGER.warning(
                        f"FFMpeg process respawn: {self.err_cnt} of {self.num_retries}")
                    return True
                else:
                    return False

    def buf_stream(self, buf_queue: Queue, stop_event: SyncEvent) -> None:
        """
        Takes video track from FFMpeg subprocess, puts in to queue.
        It is to be run in a separate process.

        Args:
            buf_queue (Queue): queue with video chunks
            stop_event (SyncEvent): set up this event to stop the method.
        """
        # Run FFMpeg in subprocess
        self.run_ffmpeg()

        # Read from pipe and put into queue
        read_size = 4096
        while not stop_event.is_set():
            try:
                bytes = self.proc.stdout.read(read_size)
                if not len(bytes):
                    # In we are here pipe is closed. It means writing end of
                    # the pipe has exited. Check we need to respawn ffmpeg.
                    if self.ffmpeg_needs_respawn():
                        self.run_ffmpeg()
                        continue
                    else:
                        break
                buf_queue.put(bytes)

            except ValueError:
                break

            except EOFError:
                continue

        buf_queue.put(None)

        buf_queue.close()
        buf_queue.join_thread()

        self.proc.terminate()

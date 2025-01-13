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
import asyncio
import copy
from enum import Enum
from multiprocessing.synchronize import Event as SyncEvent
from argparse import Namespace

LOGGER = logging.getLogger(__file__)
CHUNK_SIZE = 4096


class FFMpegProcState(Enum):
    RUNNING = 0,
    EOF = 1,
    ERROR = 2


class StreamBuffer:
    """
    Use this class to buffer video stream.
    It takes video track from input and stores it in variable size queue of bytes.
    No decoding is done, only video track demuxing. Optionally it may be dumped to HDD.
    Class may serve as buffer if your video processing code can't handle real time FPS.
    """

    def __init__(self, flags: Namespace):
        """
        Constructor

        Args:
            flags (Namespace): parsed CLI args
        """

        self.err_cnt = 0
        self.num_retries = flags.num_retries
        self.url = flags.input
        self.params = self._get_params()

        self.f_name = None
        self.f_out = None
        self.tasks = None
        self.loop = None

        if len(flags.dump):
            self.f_name = flags.dump + "." + self._format_name()
            self.loop = asyncio.get_event_loop()
            self.tasks = set()
            # Have to open and close file handle in same process.
            # So self.fout will be opened and closed in self.buf_stream

    def dump_fname(self) -> str:
        """
        Return dump filename with extension. If dump wasn't opted in, return None

        Returns:
            str: filename with extension
        """
        return self.f_name

    def chunk_size(self) -> int:
        """
        Get chunk size

        Returns:
            int: size of chunk that is put to output queue.
        """
        return CHUNK_SIZE

    def _dump(self, chunk: bytes) -> None:
        """
        Append chunk dump task to list.

        Args:
            chunk (bytes): video track bytes, will be deep copied
        """
        async def _write_chunk(f_out, chunk) -> None:
            f_out.write(chunk)

        if self.f_out:
            task = self.loop.create_task(
                _write_chunk(self.f_out, copy.deepcopy(chunk)))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.remove)

    def _await_dump(self) -> None:
        """
        Wait for all dump tasks to be completed, close the loop and file handle
        """
        if self.loop and self.tasks:
            self.loop.run_until_complete(asyncio.wait(self.tasks))
            self.loop.close()

        if self.f_out:
            self.f_out.close()

    def _get_params(self) -> Dict:
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
            self.url,
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

    def _check_up_state(self) -> tuple[bool, FFMpegProcState]:
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

    def _format_name(self) -> str:
        """
        Get output format by codec

        Returns:
            str: format name
        """

        codec = self.params["codec"]
        if codec == "h264" or codec == "hevc":
            return "mpegts"
        elif codec == "vp8" or codec == "vp9" or codec == "av1":
            return "webm"
        else:
            raise RuntimeError(f"Unsupported codec: {codec}")

    def _run_ffmpeg(self) -> None:
        """
        Run FFMpeg in subprocess and redirect output to pipe.
        """

        # Prepare FFMpeg arguments
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "quiet",
            "-i",
            self.url,
            "-map",
            "v:0",
            "-c:v",
            "copy",
            "-c:a",
            "none",
            "-f",
            self._format_name(),
            "pipe:1",
        ]

        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def _ffmpeg_needs_respawn(self) -> bool:
        """
        Check if FFMpeg process needs to be respawned.

        Returns:
            bool: True if FFMpeg needs to be respawned, False otherwise.
        """

        proc_status = self._check_up_state()
        alive = proc_status[0]
        if alive:
            return False
        else:
            death_reason = proc_status[1]
            if death_reason == FFMpegProcState.EOF:
                return False
            else:
                if self.err_cnt < self.num_retries:
                    LOGGER.warning(
                        f"FFMpeg process respawn {self.err_cnt} of {self.num_retries}")
                    return True
                else:
                    return False

    def buf_stream(self, buf_queue: Queue, stop_event: SyncEvent = None) -> None:
        """
        Takes video track from FFMpeg subprocess, puts in to queue. \\
        It is to be run in a separate process.
        Puts `None` into :arg:`buf_queue` to signal "no more data" to consumer.

        Args:
            buf_queue (Queue): queue with video chunks
            stop_event (SyncEvent): set up to stop, otherwise will run till EOF
        """
        # Need to open and close file handle in the same process, so do it here
        if self.f_name:
            self.f_out = open(self.f_name, "ab")

        # Run ffmpeg in subprocess
        self._run_ffmpeg()

        # Read from pipe and put into queue
        while not stop_event.is_set() if stop_event else True:
            try:
                bytes = self.proc.stdout.read(CHUNK_SIZE)
                if not len(bytes):
                    if self._ffmpeg_needs_respawn():
                        self._run_ffmpeg()
                        continue
                    else:
                        break
                buf_queue.put(bytes)
                self._dump(bytes)

            except ValueError:
                break

            except EOFError:
                continue

        buf_queue.put(None)
        buf_queue.close()

        # Wait for dump to complete and close file handle
        self._await_dump()
        self.proc.terminate()

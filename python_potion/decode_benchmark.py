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

import python_potion.buffering as buffering
import python_potion.decoder as decoder
from argparse import Namespace
import time
import os
import concurrent.futures
import logging

from queue import Empty
from multiprocessing import Queue, Process


LOGGER = logging.getLogger(__file__)


class DecodeBenchmark:
    """
    Decoding performance benchmark.
    Takes path to file with list of videos.
    Every video will be decoded in separate thread.
    After all videos are decoded, will output short stat to stdout.
    """

    def __init__(self, flags: Namespace):
        self.flags = flags
        self.gpu_id = flags.gpu_id

        with open(flags.input) as inp_file:
            if not os.path.isfile(flags.input):
                raise Exception(f"{flags.input} is not a file")
            self.input_files = [line.rstrip() for line in inp_file]

        self.res_size = len(self.input_files)
        self.results = Queue(maxsize=self.res_size)

    def run(self) -> None:
        """
        Run multiple decoding threads.
        Wait until done, output short statistics to stdout.
        """

        try:
            tasks = set()
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.input_files))

            for input in self.input_files:
                future = executor.submit(self._decode_func, input)
                tasks.add(future)
                future.add_done_callback(tasks.remove)

            while len(tasks):
                time.sleep(1)

            while True:
                try:
                    result = self.results.get_nowait()
                    url = result[0]
                    cnt = result[1]
                    fps = cnt / result[2]
                    print(f"url: {url}, frames: {cnt}, fps: {fps}")
                except Empty:
                    break

        except Exception as e:
            LOGGER.fatal(str(e))

    def _decode_func(self, input: str) -> None:
        """
        Decode single video.
        To be run in separate thread.

        Args:
            input (str): input video URL.
        """

        try:
            dec_frames = 0
            start = time.time()

            my_flags = self.flags
            my_flags.input = input
            buf = buffering.StreamBuffer(my_flags)

            buf_queue = Queue(maxsize=my_flags.buf_queue_size)

            buf_proc = Process(
                target=buf.bufferize,
                args=(buf_queue, None),
            )

            buf_proc.start()

            dec = decoder.Decoder(buf_queue, self.flags)
            while True:
                surf = dec.decode(dry_run=True)
                if not surf:
                    break
                dec_frames += 1

            buf_proc.join()
            self.results.put((input, dec_frames, time.time() - start))

        except Exception as e:
            LOGGER.fatal(str(e))

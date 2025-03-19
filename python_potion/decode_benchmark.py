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

from multiprocessing import Queue, Process


LOGGER = logging.getLogger(__file__)


class DecodeBenchmark:
    def __init__(self, flags: Namespace):
        self.flags = flags
        self.gpu_id = flags.gpu_id
        self.num_procs = flags.num_procs

        with open(flags.input) as inp_file:
            if not os.path.isfile(inp_file):
                raise Exception(f"{inp_file} is not a file")
            self.input_files = [line.rstrip() for line in inp_file]

    def run(self) -> None:
        try:
            tasks = set()
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_procs)

            for _ in range(0, self.num_procs):
                results = []
                future = executor.submit(self._spawn_threads, results)
                tasks.add(future)
                future.add_done_callback(tasks.remove)

            while len(tasks):
                time.sleep(1)

            if len(results) != len(self.input_files):
                LOGGER.error(f"Not all input files were processed")

            for i in range(0, len(results)):
                num_frames = results[i][0]
                run_time = results[i][1]
                print(
                    f"{self.input_files[i]} : {num_frames} frames in {run_time} s.")

        except Exception as e:
            LOGGER.fatal(str(e))

    def _spawn_threads(self, results: list) -> None:
        try:
            tasks = set()
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.input_files))

            for input in self.input_files:
                future = executor.submit(self._decode_func, input, results)
                tasks.add(future)
                future.add_done_callback(tasks.remove)

            while len(tasks):
                time.sleep(1)

        except Exception as e:
            LOGGER.fatal(str(e))

    def _decode_func(self, input: str, results: list) -> None:
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
                surf = dec.decode()
                if not surf:
                    break
                dec_frames += 1

            buf_proc.join()
            results.append((dec_frames, time.time() - start))

        except Exception as e:
            LOGGER.fatal(str(e))

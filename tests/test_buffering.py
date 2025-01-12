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

import json
import time
import unittest
import test_common as tc
import multiprocessing as mp
import python_potion.buffering as buffering

from multiprocessing import Queue, Process
from queue import Empty

GT_FILENAME = "gt_files.json"
TEST_CASE = "basic"
MB = 1024 * 1024
THRESHOLD = 0.05


class TestLocal(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)

        with open(GT_FILENAME, "r") as f_in:
            self.gt = json.load(f_in)[TEST_CASE]

        self.flags = tc.make_flags(input=self.gt["uri"])

    def test__get_params(self):
        """
        This test checks URL probe
        """

        buf = buffering.StreamBuffer(self.flags)

        params = buf._get_params()
        for key in params.keys():
            self.assertEqual(str(params[key]), str(self.gt[key]))

    def test_format_by_codec(self):
        """
        This test checks if format name is generated as expected.
        """
        buf = buffering.StreamBuffer(self.flags)
        self.assertEqual(buf.format_by_codec(),
                         self.gt["buf_fmt_name"]["value"])

    def test_buf_stream_till_eof(self):
        """
        This test checks that bufferization processes the whole input
        """

        buf = buffering.StreamBuffer(self.flags)
        buf_queue = Queue(maxsize=0)
        buf_proc = Process(
            target=buf.buf_stream,
            args=(buf_queue, None),
        )

        buf_proc.start()
        buf_proc.join()

        file_size = float(self.gt["filesize"]) / MB
        chunk_size = float(buf_queue.qsize() * buf.chunk_size()) / MB

        # Size mismatch within 5% tolerance is considered OK.
        self.assertLessEqual(
            abs(file_size - chunk_size) / file_size, THRESHOLD)

    def test_buf_stream_till_stop(self):
        """
        This test checks the bufferization process stops at event
        """

        buf = buffering.StreamBuffer(self.flags)
        buf_queue = Queue(maxsize=0)
        buf_proc_stop = mp.Event()
        buf_proc = Process(
            target=buf.buf_stream,
            args=(buf_queue, buf_proc_stop),
        )

        # Can't guarantee what amount of input will be processed when process is stopped.
        # So if the test finishes it's considered to be success.
        buf_proc.start()
        buf_proc_stop.set()
        buf_proc.join()

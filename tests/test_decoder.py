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

import os
import json
import unittest
import test_common as tc
import multiprocessing as mp
import python_potion.buffering as buffering
import python_potion.decoder as decoder

from multiprocessing import Queue, Process
from pathlib import Path
from parameterized import parameterized

GT_FILENAME = "gt_files.json"
TEST_CASE = "basic"
MB = 1024 * 1024
THRESHOLD = 0.05


class TestOnLocalStream(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)

        with open(GT_FILENAME, "r") as f_in:
            self.gt = json.load(f_in)[TEST_CASE]

        self.flags = tc.make_flags(input=self.gt["uri"])

    def test_num_frames(self):
        """
        Check if all video frames are decoded
        """
        buf = buffering.StreamBuffer(self.flags)
        buf_queue = Queue(maxsize=0)
        buf_proc = Process(
            target=buf.buf_stream,
            args=(buf_queue, None),
        )

        buf_proc.start()
        buf_proc.join()

        dec = decoder.NvDecoder(buf_queue, self.flags.gpu_id)
        dec_cnt = 0
        while True:
            surf = dec.decode()
            if not surf:
                break
            dec_cnt += 1

        self.assertEqual(dec_cnt, self.gt["num_frames"])

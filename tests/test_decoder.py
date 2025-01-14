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
import unittest
import test_common as tc
import python_potion.buffering as buffering
import python_potion.decoder as decoder

from multiprocessing import Queue, Process

GT_FILENAME = "gt_files.json"
TEST_CASE = "basic"


class TestDecoder(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)

        with open(GT_FILENAME, "r") as f_in:
            self.gt = json.load(f_in)[TEST_CASE]

        self.flags = tc.make_flags(input=self.gt["uri"])

    def test_check_surf(self):
        """
        Check that surfaces have proper dimensions and format
        """
        buf = buffering.StreamBuffer(self.flags)
        buf_queue = Queue(maxsize=0)
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

            self.assertEqual(surf.Width, self.gt["width"])
            self.assertEqual(surf.Height, self.gt["height"])
            self.assertEqual(str(surf.Format), self.gt["format"])

        buf_proc.join()

    def test_num_frames(self):
        """
        Check if all video frames are decoded
        """
        buf = buffering.StreamBuffer(self.flags)
        buf_queue = Queue(maxsize=0)
        buf_proc = Process(
            target=buf.bufferize,
            args=(buf_queue, None),
        )

        buf_proc.start()

        dec = decoder.Decoder(buf_queue, self.flags)
        dec_cnt = 0
        while dec.decode() is not None:
            dec_cnt += 1

        self.assertEqual(dec_cnt, self.gt["num_frames"])
        buf_proc.join()

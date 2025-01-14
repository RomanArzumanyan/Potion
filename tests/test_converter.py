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
import python_potion.converter as converter
import python_vali as vali

from multiprocessing import Queue, Process

GT_FILENAME = "gt_files.json"
TEST_CASE = "basic"
THRESHOLD = 0.05


class TestConverter(unittest.TestCase):
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

        # Most common VGG input params.
        params = {
            "src_fmt": dec.format(),
            "dst_fmt": vali.PixelFormat.RGB_32F_PLANAR,
            "src_w": dec.width(),
            "src_h": dec.height(),
            "dst_w": 224,
            "dst_h": 224
        }
        cvt = converter.Converter(params, self.flags)

        while True:
            surf_src = dec.decode()
            if not surf_src:
                break

            surf_dst = cvt.convert(surf_src)
            if not surf_dst:
                break

            self.assertEqual(surf_dst.Width, params["dst_w"])
            self.assertEqual(surf_dst.Height, params["dst_h"])
            self.assertEqual(surf_dst.Format, params["dst_fmt"])

        buf_proc.join()

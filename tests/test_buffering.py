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

GT_FILENAME = "gt_files.json"
TEST_CASE = "basic"


class TestLocal(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)

    def test__get_params(self):
        """
        This test checks URL probe
        """

        with open(GT_FILENAME, "r") as f_in:
            gt = json.load(f_in)[TEST_CASE]

        flags = tc.make_flags(input=gt["uri"])
        buf = buffering.StreamBuffer(flags)

        params = buf._get_params()
        for key in params.keys():
            self.assertEqual(str(params[key]), str(gt[key]))

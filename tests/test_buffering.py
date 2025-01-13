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

from multiprocessing import Queue, Process
from pathlib import Path
from parameterized import parameterized
from queue import Empty

GT_FILENAME = "gt_files.json"
TEST_CASE = "basic"
THRESHOLD = 0.05


def drain(q: Queue) -> int:
    """
    Drain queue. Will not return unless :arg:`q` is empty.

    Args:
        q (Queue): input queue.

    Returns:
        int: drained data size.
    """
    size = 0
    while True:
        try:
            chunk = q.get_nowait()
            if chunk is None:
                return size
            size += len(chunk)

        except Empty:
            continue

        except ValueError:
            break
    return size


class TestOnLocalStream(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName)

        with open(GT_FILENAME, "r") as f_in:
            self.gt = json.load(f_in)[TEST_CASE]

        self.flags = tc.make_flags(input=self.gt["uri"])

    def test__get_params(self):
        """
        This test checks URL probe.

        _get_params is a 'private' method, but the video track params
        are single most important set of parameters, so we'd like to 
        test it.
        """

        buf = buffering.StreamBuffer(self.flags)

        params = buf._get_params()
        for key in params.keys():
            self.assertEqual(str(params[key]), str(self.gt[key]))

    def test__format_name(self):
        """
        This test checks if format name is generated as expected.

        _format_name is a 'private' method, but it influences dump file
        extension which is visible to user, hence we want to test it.
        """
        buf = buffering.StreamBuffer(self.flags)
        self.assertEqual(buf._format_name(),
                         self.gt["buf_fmt_name"]["value"])

    def test_invalid_url(self):
        """
        Test invalid input url.
        """
        flags = self.flags
        flags.input = "nonexistent.mp4"
        try:
            buf = buffering.StreamBuffer(flags)

        except ValueError as e:
            err_str = 'No stream parameters found'
            self.assertRegex(str(e), err_str)
            return

        self.fail("Test is expected to raise ValueError exception")

    def test_buf_stream_till_eof(self):
        """
        Check if bufferization processes the whole input.
        """

        buf = buffering.StreamBuffer(self.flags)
        buf_queue = Queue(maxsize=0)
        buf_proc = Process(
            target=buf.buf_stream,
            args=(buf_queue, None),
        )

        buf_proc.start()
        chunk_size = drain(buf_queue)
        buf_proc.join()

        file_size = float(self.gt["filesize"])
        self.assertLessEqual(
            abs(file_size - chunk_size) / file_size, THRESHOLD)

    @parameterized.expand([
        ["video_track"],
        [""]
    ])
    def test_dump(self, dump_fname: str):
        """
        Test video track dump feature.
        """
        flags = self.flags
        flags.dump = dump_fname

        buf = buffering.StreamBuffer(flags)
        buf_queue = Queue(maxsize=0)
        buf_proc = Process(
            target=buf.buf_stream,
            args=(buf_queue, None),
        )

        buf_proc.start()
        drain(buf_queue)
        buf_proc.join()

        if not len(dump_fname):
            self.assertIsNone(buf.dump_fname())
        else:
            fname = Path(buf.dump_fname())
            self.assertEqual(True, fname.is_file())
            self.assertGreater(os.path.getsize(fname), 0)
            os.remove(fname)

    def test_buf_stream_till_stop(self):
        """
        Check if bufferization process stops at event.
        """

        buf = buffering.StreamBuffer(self.flags)
        buf_queue = Queue(maxsize=0)
        buf_proc_stop = mp.Event()
        buf_proc = Process(
            target=buf.buf_stream,
            args=(buf_queue, buf_proc_stop),
        )

        # Can't guarantee what amount of input will be processed when process
        # is stopped. So if the test finishes it's considered success.
        buf_proc.start()
        buf_proc_stop.set()
        drain(buf_queue)
        buf_proc.join()

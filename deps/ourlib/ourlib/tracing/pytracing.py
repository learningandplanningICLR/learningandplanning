# -*- coding: utf-8 -*-
# COPYRIGHT(): copied/based on https://github.com/kwlzn/pytracing
import os
import sys
import json
import time
import threading
from contextlib import contextmanager

from queue import Queue


def to_microseconds(s):
    return 1000000 * float(s)


class TraceWriter(threading.Thread):

    def __init__(self, terminator, input_queue, output_stream):
        threading.Thread.__init__(self)
        self.daemon = True
        self.terminator = terminator
        self.input = input_queue
        self.output = output_stream

    def run(self):
        while not self.terminator.is_set():
            item = self.input.get()
            self.output.write(item.encode())


# COPIED from https://stackoverflow.com/questions/34115298/how-do-i-get-the-current-depth-of-the-python-interpreter-stack
def get_stack_size():
    """Get stack size for caller's frame.

    %timeit len(inspect.stack())
    8.86 ms ± 42.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    %timeit get_stack_size()
    4.17 µs ± 11.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    """
    size = 2  # current frame and caller's frame always exist
    while True:
        try:
            sys._getframe(size)
            size += 1
        except ValueError:
            return size - 1  # subtract current frame


class TraceProfiler(object):
    """A python trace profiler that outputs Chrome Trace-Viewer format (about://tracing).

     Usage:

        from pytracing import TraceProfiler
        tp = TraceProfiler(output=open('/tmp/trace.out', 'wb'))
        with tp.traced():
          ...

  """
    TYPES = {'call': 'B', 'return': 'E'}

    def __init__(self, output, clock=None, max_depth=6, tid_prefix=''):
        '''

        :param output:
        :param clock:
        :param max_depth:
        :param tid_prefix: This can be used to add information about rank of the process(when using
        MPI)
        '''
        self.output = output
        self.clock = clock or time.time
        self.pid = os.getpid()
        self.queue = Queue()
        self.terminator = threading.Event()
        self.writer = TraceWriter(self.terminator, self.queue, self.output)
        self.base_depth = None
        self.max_depth = max_depth
        self.tid_prefix = tid_prefix

    @property
    def thread_id(self):
        return threading.current_thread().name

    @contextmanager
    def traced(self):
        """Context manager for install/shutdown in a with block."""
        self.base_depth = get_stack_size()
        self.install()
        try:
            yield
        finally:
            self.shutdown()

    def install(self):
        """Install the trace function and open the JSON output stream."""
        self._open_collection()  # Open the JSON output.
        self.writer.start()  # Start the writer thread.
        sys.setprofile(self.tracer)  # Set the trace/profile function.
        threading.setprofile(self.tracer)  # Set the trace/profile function for threads.

    def shutdown(self):
        sys.setprofile(None)  # Clear the trace/profile function.
        threading.setprofile(None)  # Clear the trace/profile function for threads.
        self._close_collection()  # Close the JSON output.
        self.terminator.set()  # Stop the writer thread.
        self.writer.join()  # Join the writer thread.

    def _open_collection(self):
        """Write the opening of a JSON array to the output."""
        self.queue.put('[\n')

    def _close_collection(self):
        """Write the closing of a JSON array to the output."""
        self.queue.put('{}\n]\n')

    def fire_event(self, event_type, func_name, func_filename, func_line_no,
                   caller_filename, caller_line_no):
        """Write a trace event to the output stream."""
        timestamp = to_microseconds(self.clock())
        # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
        event = json.dumps(
            dict(
                name=func_name,  # Event Name.
                cat=func_filename,  # Event Category.
                tid=self.tid_prefix + self.thread_id,  # Thread ID.
                ph=self.TYPES[event_type],  # Event Type.
                pid=self.pid,  # Process ID.
                ts=timestamp,  # Timestamp.
                args=dict(
                    function=':'.join([str(x) for x in (func_filename, func_line_no, func_name)]),
                    caller=':'.join([str(x) for x in (caller_filename, caller_line_no)]),
                )
            )
        ) + ',\n'
        self.queue.put(event)

    def tracer(self, frame, event_type, arg):
        """Bound tracer function for sys.settrace()."""
        try:
            if event_type in self.TYPES.keys() and frame.f_code.co_name != 'write':
                if get_stack_size() <= self.base_depth + self.max_depth:
                    self.fire_event(
                        event_type=event_type,
                        func_name=frame.f_code.co_name,
                        func_filename=frame.f_code.co_filename,
                        func_line_no=frame.f_lineno,
                        caller_filename=frame.f_back.f_code.co_filename,
                        caller_line_no=frame.f_back.f_lineno,
                    )
        except Exception:
            pass  # Don't disturb execution if we can't log the trace.

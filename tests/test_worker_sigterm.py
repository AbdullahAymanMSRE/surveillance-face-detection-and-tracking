"""The worker must shut down gracefully on SIGTERM so in-flight clips upload.

The supervisor stops a worker with ``proc.terminate()`` (SIGTERM). By default
SIGTERM kills the process without running the main loop's ``finally`` block, so a
visit open at stop-time loses its buffered clip and is left dangling for the
reaper. ``_install_sigterm_handler`` converts SIGTERM into a ``KeyboardInterrupt``
the loop catches, letting the cleanup run. (The full end-to-end path is covered
by a manual file-source repro documented in the fix commit.)
"""

import signal

import pytest

import pipeline_node


def test_sigterm_handler_raises_keyboardinterrupt():
    previous = signal.getsignal(signal.SIGTERM)
    try:
        pipeline_node._install_sigterm_handler()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        # Invoking the installed handler must raise, so it propagates out of the
        # worker's try/while and triggers the cleanup `finally`.
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)

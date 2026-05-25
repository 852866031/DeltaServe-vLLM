# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DeltaServe co-serving additions on top of vLLM (net-new on our fork).

Everything we add for inference + LoRA finetuning co-serving lives under this
package so it stays localized and easy to rebase. Phase 1 starts with a small
logging helper so all our runtime prints are easy to spot in vLLM's verbose
startup output.
"""

import sys

_GREEN = "\033[92m"   # main process (engine / worker)
_PURPLE = "\033[95m"  # backward subprocess
_RESET = "\033[0m"

# Set True inside the spawned backward process so its prints are purple. The
# main process leaves it False (green). Distinguishing by color makes it obvious
# which process a line came from when their stdout interleaves.
_IS_BACKWARD = False


def mark_backward_process() -> None:
    """Called once at the start of the backward child so dprint goes purple."""
    global _IS_BACKWARD
    _IS_BACKWARD = True


def dprint(msg: str, **kwargs) -> None:
    """Print a [deltaserve]-prefixed line: green in the main process, purple in
    the backward subprocess.

    Color is only emitted when stdout is a TTY, so piping to a file or through
    a log collector stays clean. Always flushes (these are multi-process
    startup/handshake signals we want to see immediately).
    """
    stream = kwargs.pop("file", sys.stdout)
    color = _PURPLE if _IS_BACKWARD else _GREEN
    prefix = "[deltaserve] "
    if hasattr(stream, "isatty") and stream.isatty():
        text = f"{color}{prefix}{msg}{_RESET}"
    else:
        text = f"{prefix}{msg}"
    print(text, file=stream, flush=True, **kwargs)

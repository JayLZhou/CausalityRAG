"""Run an evaluation script with a longer ReaderClient HTTP timeout.

Usage: python run_with_reader_timeout.py SECONDS SCRIPT [ARGS...]
Only the client-side wait changes; decoding (no max_tokens cap) is untouched,
so a reader that generates to the context limit still gets its full answer.
"""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())  # the runner cds into the code repo
import causalityrag.reader as reader  # noqa: E402

seconds = int(sys.argv[1])
_init = reader.ReaderClient.__init__


def _init_with_timeout(self, *args, **kwargs):
    kwargs.setdefault("timeout", seconds)
    _init(self, *args, **kwargs)


reader.ReaderClient.__init__ = _init_with_timeout

if os.environ.get("LENIENT_QUARTZ"):
    # Three QuaRTz clean answers map to no answer choice; Table 3 kept those queries
    # as valid, so compare their raw strings instead of raising.
    _label = reader.quartz_choice_label

    def _lenient_label(question, answer):
        try:
            return _label(question, answer)
        except reader.ReaderProtocolError:
            return str(answer).strip().casefold()

    reader.quartz_choice_label = _lenient_label
    _quartz_canon = reader.canonicalize_quartz_answer

    def _lenient_quartz_canon(question, answer):
        try:
            return _quartz_canon(question, answer)
        except (reader.ReaderProtocolError, KeyError):
            return str(answer).strip()

    reader.canonicalize_quartz_answer = _lenient_quartz_canon
    _canon = reader.canonicalize_reader_answer

    def _lenient_canon(question, answer, *, reader_mode):
        try:
            return _canon(question, answer, reader_mode=reader_mode)
        except (reader.ReaderProtocolError, KeyError):
            return str(answer).strip()

    reader.canonicalize_reader_answer = _lenient_canon
script = sys.argv[2]
sys.argv = sys.argv[2:]
runpy.run_path(script, run_name="__main__")

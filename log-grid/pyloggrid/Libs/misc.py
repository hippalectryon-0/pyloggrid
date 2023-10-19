"""Misc small libraries that help with various tasks, too small to fit in their own library file"""
import time

import numpy as np


def composed_decs(*decs):
    """Compose several decorators into one.

    Example:
        ::

            @composed(dec_1, dec_2)
            def f(...)

        is equivalent to

        ::

            @dec_1
            @dec_2
            def f(...)
    """

    # noinspection PyMissingOrEmptyDocstring
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def bytes2human(n, format_="%(value).1f %(symbol)s", symbols="customary"):
    """Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: https://goo.gl/kTQMs

    Example:
        ::

            > bytes2human(0)
            '0.0 B'
            > bytes2human(0.9)
            '0.0 B'
            > bytes2human(1)
            '1.0 B'
            > bytes2human(1.9)
            '1.0 B'
            > bytes2human(1024)
            '1.0 K'
            > bytes2human(1048576)
            '1.0 M'
            > bytes2human(1099511627776127398123789121)
            '909.5 Y'

            > bytes2human(9856, symbols="customary")
            '9.6 K'
            > bytes2human(9856, symbols="customary_ext")
            '9.6 kilo'
            > bytes2human(9856, symbols="iec")
            '9.6 Ki'
            > bytes2human(9856, symbols="iec_ext")
            '9.6 kibi'

            > bytes2human(10000, "%(value).1f %(symbol)s/sec")
            '9.8 K/sec'

            > # precision can be adjusted by playing with %f operator
            > bytes2human(10000, format_="%(value).5f %(symbol)s")
            '9.76562 K'
    """
    SYMBOLS = {
        "customary": ("B", "K", "M", "G", "T", "P", "E", "Z", "Y"),
        "customary_ext": ("byte", "kilo", "mega", "giga", "tera", "peta", "exa", "zetta", "iotta"),
        "iec": ("Bi", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"),
        "iec_ext": ("byte", "kibi", "mebi", "gibi", "tebi", "pebi", "exbi", "zebi", "yobi"),
    }

    n = int(n)
    assert n >= 0
    symbols = SYMBOLS[symbols]
    prefix = {s: 1 << (i + 1) * 10 for i, s in enumerate(symbols[1:])}
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format_ % locals()
    return format_ % dict(symbol=symbols[0], value=n)


class TimeTracker:
    """Keep track of elapsed time. Supports nesting.

    How to use:

    * to start a timer, use ``timer.start_timer(key)`` where ``key`` is a unique identifier
    * to stop a timer and add the elapsed time to the total for the key, use ``timer.stop_timer(key)``
    * to stop and immediately restart (used for storing the current value) a timer use ``timer.tick_timer(key)``

    Wherever possible, prefer wrapping you code in a with block:
    ::

        with timer(key):
            ...

    """

    def __init__(self, initdict: dict = None):
        self.elapsed_time = initdict or {}  # total elapsed time
        self.timer_start = {}  # start of each tracker
        self.current_key = []  # list of current keys to enable nesting with blocks

    def __call__(self, key: str):
        self.current_key.append(key)
        return self

    def __enter__(self):
        self.start_timer(self.current_key[-1])

    def __exit__(self, *args):
        self.end_timer(self.current_key.pop())

    def get(self, key: str) -> float:
        """Get elapsed time for ``key``."""
        return self.elapsed_time.get(key, 0)

    def start_timer(self, key: str) -> None:
        """Start timer for ``key``"""
        self.timer_start[key] = time.perf_counter()

    def end_timer(self, key: str) -> float:
        """Stop timer for ``key>``, store and return total elapsed time."""
        elapsed = time.perf_counter() - self.timer_start[key]
        if key not in self.elapsed_time:
            self.elapsed_time[key] = 0
        self.elapsed_time[key] += elapsed
        return self.elapsed_time[key]

    def tick_timer(self, key: str) -> None:
        """Store elapsed time and reset timer."""
        self.end_timer(key)
        self.start_timer(key)

    def __repr__(self):
        def format_time(seconds: float) -> str:
            """human-readable"""
            # noinspection PyUnresolvedReferences
            delta_dict = {
                "d": seconds // (3600 * 24),
                "h": seconds // 3600,
                "m": seconds // 60,
                "s": seconds // 1,
                "ms": np.round(1000 * (seconds % 1), 3).astype(int),
            }
            res = " ".join(f"{int(v)}{k}" for k, v in delta_dict.items() if v)
            return res or "0ms"

        return "{" + ", ".join([f"{k}: {format_time(v)}" for k, v in sorted(self.elapsed_time.items(), key=lambda item: -item[1])]) + "}"

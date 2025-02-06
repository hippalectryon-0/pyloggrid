"""utests for misc.py"""

import time

import numpy as np

from pyloggrid.Libs.misc import TimeTracker, bytes2human, composed_decs


class TestBytes2Human:
    def test_zero_bytes(self):
        assert bytes2human(0) == "0.0 B"

    def test_sub_byte(self):
        assert bytes2human(0.9) == "0.0 B"

    def test_single_byte(self):
        assert bytes2human(1) == "1.0 B"

    def test_less_than_one_kilobyte(self):
        assert bytes2human(1.9) == "1.0 B"

    def test_one_kilobyte(self):
        assert bytes2human(1024) == "1.0 K"

    def test_one_megabyte(self):
        assert bytes2human(1048576) == "1.0 M"

    def test_very_large_number(self):
        assert bytes2human(1099511627776127398123789121) == "909.5 Y"

    def test_customary_symbols(self):
        assert bytes2human(9856, symbols="customary") == "9.6 K"

    def test_customary_ext_symbols(self):
        assert bytes2human(9856, symbols="customary_ext") == "9.6 kilo"

    def test_iec_symbols(self):
        assert bytes2human(9856, symbols="iec") == "9.6 Ki"

    def test_iec_ext_symbols(self):
        assert bytes2human(9856, symbols="iec_ext") == "9.6 kibi"

    def test_custom_format(self):
        assert bytes2human(10000, "%(value).1f %(symbol)s/sec") == "9.8 K/sec"

    def test_precision(self):
        assert bytes2human(10000, format_="%(value).5f %(symbol)s") == "9.76562 K"


class TestTimeTracker:
    def test_tick_timer(self):
        timer = TimeTracker()
        timer.start_timer("key1")
        time.sleep(0.1)
        timer.tick_timer("key1")
        elapsed1 = timer.get("key1")
        time.sleep(0.1)
        elapsed2 = timer.end_timer("key1")
        assert np.isclose(elapsed1, 0.1, rtol=0.1)
        assert np.isclose(elapsed2, 0.2, rtol=0.1)

    def test_time_tracker(self):
        # Create TimeTracker instance
        tt = TimeTracker()

        # Start timer for key "a"
        tt.start_timer("a")
        time.sleep(0.5)
        elapsed_a = tt.end_timer("a")

        # Start timer for key "b"
        tt.start_timer("b")
        time.sleep(0.5)
        elapsed_b = tt.end_timer("b")

        # Verify elapsed time for "a" and "b"
        assert np.isclose(elapsed_a, 0.5, rtol=0.1)
        assert np.isclose(elapsed_b, 0.5, rtol=0.1)

        # Verify total elapsed time for "a" and "b"
        assert elapsed_a == tt.get("a")
        assert elapsed_b == tt.get("b")

        # Test with block
        with tt("c"):
            time.sleep(0.5)
        assert np.isclose(tt.get("c"), 0.5, rtol=0.1)

        # Test nested with blocks
        with tt("d"):
            with tt("e"):
                time.sleep(0.3)
            with tt("f"):
                time.sleep(0.5)
        assert np.isclose(tt.get("d"), 0.8, rtol=0.1)
        assert np.isclose(tt.get("e"), 0.3, rtol=0.1)
        assert np.isclose(tt.get("f"), 0.5, rtol=0.1)


class TestComposedDecs:
    @staticmethod
    def decorator_one(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs) + 1

        return wrapper

    @staticmethod
    def decorator_two(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs) * 2

        return wrapper

    @staticmethod
    def decorator_three(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs) - 3

        return wrapper

    def test_composed_decs(self):
        # noinspection PyMissingOrEmptyDocstring
        @composed_decs(TestComposedDecs.decorator_one, TestComposedDecs.decorator_two, TestComposedDecs.decorator_three)
        def my_func_1_2_3(x):
            return x

        # noinspection PyMissingOrEmptyDocstring
        @composed_decs(TestComposedDecs.decorator_one, TestComposedDecs.decorator_two)
        def my_func_1_2(x):
            return x

        assert my_func_1_2(2) == 5
        assert my_func_1_2_3(2) == -1

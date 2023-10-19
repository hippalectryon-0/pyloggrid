"""utests for custom_logger.py"""
import logging
import os.path
import re

from pyloggrid.Libs.custom_logger import setup_custom_logger


def test_IO(tmp_path):
    """test logger"""

    name = "TESTNAME"
    level = logging.INFO
    outfile = os.path.join(tmp_path, "test.out")

    logger = setup_custom_logger(name=name, level=level, logfile=outfile)
    logger.info("TEST INFO")
    logger.debug("TEST DEBUG")

    # check if log file exists
    assert os.path.isfile(outfile), "No log file found"
    with open(outfile, "r") as f:
        logtxt = f.read()

    # assert proper logging
    assert re.match(r"^([0-9\-]+)\s([0-9:]+)\sINFO\s+TESTNAME\s+TEST INFO", logtxt), "INFO line not logged / wrong format"
    assert "TEST DEBUG" not in logtxt, "Logging levels not excluding properly"

    # Setup a new logger and make sure we don't log twice
    logger = setup_custom_logger(name=name, level=level, logfile=outfile)
    logger.info("TEST DUPLICATE")

    with open(outfile, "r") as f:
        logtxt = f.read()

    res = re.findall("TEST DUPLICATE", logtxt)
    assert res is not None and len(res) == 1

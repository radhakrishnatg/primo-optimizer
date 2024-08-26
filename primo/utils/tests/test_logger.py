#################################################################################
# PRIMO - The P&A Project Optimizer was produced under the Methane Emissions
# Reduction Program (MERP) and National Energy Technology Laboratory's (NETL)
# National Emissions Reduction Initiative (NEMRI).
#
# NOTICE. This Software was developed under funding from the U.S. Government
# and the U.S. Government consequently retains certain rights. As such, the
# U.S. Government has been granted for itself and others acting on its behalf
# a paid-up, nonexclusive, irrevocable, worldwide license in the Software to
# reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit others to do so.
#################################################################################

# Standard libs
import logging
import os

# Installed libs
import pytest

# User-defined libs
from primo.utils import setup_logger, LogLevel

LOGGER = logging.getLogger(__name__)


# pylint: disable = unspecified-encoding
def dummy_func():
    """A dummy function that logs a few messages"""
    LOGGER.info("Beginning")
    LOGGER.info("Middle")
    LOGGER.warning("Careful!")
    LOGGER.info("Ending")


def test_logger():
    """Tests if the setup_logger function works correctly or not"""
    setup_logger(
        log_level=LogLevel.INFO,
        log_to_console=True,
        log_file="mylog.log",
    )
    dummy_func()

    assert os.path.exists("mylog.log")

    with open("mylog.log", "r") as fp:
        data = fp.read()

    assert "primo.utils.tests.test_logger:30 - INFO - Beginning" in data
    assert "primo.utils.tests.test_logger:31 - INFO - Middle" in data
    assert "primo.utils.tests.test_logger:32 - WARNING - Careful!" in data
    assert "primo.utils.tests.test_logger:33 - INFO - Ending" in data

    # Catch the log-file-exists error
    with pytest.raises(
        ValueError,
        match=("Log file: mylog.log already exists. Please specify new log file."),
    ):
        setup_logger(
            log_level=LogLevel.WARNING,
            log_to_console=True,
            log_file="mylog.log",
        )

    # Now, delete the log file and try it again
    os.remove("mylog.log")
    assert not os.path.exists("mylog.log")

    setup_logger(
        log_level=LogLevel.WARNING,
        log_to_console=True,
        log_file="mylog.log",
    )

    # Run the dummy function
    dummy_func()

    assert os.path.exists("mylog.log")
    with open("mylog.log", "r") as fp:
        data = fp.read()

    # Now, INFO-level messages should not have been printed
    assert "primo.utils.tests.test_logger:30 - INFO - Beginning" not in data
    assert "primo.utils.tests.test_logger:31 - INFO - Middle" not in data
    assert "primo.utils.tests.test_logger:32 - WARNING - Careful!" in data
    assert "primo.utils.tests.test_logger:33 - INFO - Ending" not in data

    setup_logger(
        log_level=LogLevel.WARNING,
        log_to_console=False,
    )

    os.remove("mylog.log")

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
import enum
import logging
import os
import pathlib
import sys

# Installed libs

# User defined libs
from primo.utils.raise_exception import raise_exception

LOGGER_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
LOGGER_DATE = "%d-%b-%y %H:%M:%S"


class LogLevel(enum.Enum):
    """Enum for logging levels"""

    CRITICAL = logging.CRITICAL
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


def setup_logger(
    log_level: LogLevel = LogLevel.INFO,
    log_to_console: bool = True,
    log_file: pathlib.Path = pathlib.Path(os.devnull),
):
    """
    Set up logging objects based on user input.

    Parameters
    ----------
    log_level : primo.utils.setup_logger.LogLevel, default = LogLevel.INFO
        Levels of Logging: CRITICAL, WARNING, INFO, DEBUG
    log_to_console : bool, default = True
        If True, log messages are displayed on the screen in addition
        to the log file (if configured)
    log_file : pathlib.Path, default = pathlib.Path(os.devnull)
        The path on the disk where log files are written

    Returns
    -------
    logging.Logger
        A logger object set up as required

    Raises
    ------
    ValueError
        If the log_file specified already exists or if an invalid value for
        log_level is provided
    """

    # If there are any existing handlers, remove them
    # This is needed to update the logging level if a
    # logger has already been set.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = []
    if log_to_console:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)

    if log_file != pathlib.Path(os.devnull):
        if os.path.exists(log_file):
            raise_exception(
                f"Log file: {log_file} already exists. Please specify new log file.",
                ValueError,
            )
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level.value,
        format=LOGGER_FORMAT,
        datefmt=LOGGER_DATE,
        handlers=handlers,
    )

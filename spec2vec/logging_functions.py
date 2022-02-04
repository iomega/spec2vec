"""Spec2Vec logger.

Spec2Vec functions and method report unexpected or undesired behavior as
logging WARNING, and additional information as INFO.
The default logging level is set to WARNING.
The logger is an adaptation of the matchms logger.


If you want to output additional
logging messages, you can lower the logging level to INFO using set_spec2vec_logger_level:

.. code-block:: python

    from spec2vec import set_spec2vec_logger_level

    set_spec2vec_logger_level("INFO")

This can also be combined with setting the matchms logger which occurs separately
by using set_matchms_logger_level:

.. code-block:: python

    from matchms import set_matchms_logger_level
    from spec2vec import set_spec2vec_logger_level

    set_matchms_logger_level("INFO")
    set_spec2vec_logger_level("INFO")

If you want to suppress logging warnings, you can also raise the logging level
to ERROR by:

.. code-block:: python

    set_spec2vec_logger_level("ERROR")

To write logging entries to a local file, you can do the following:

.. code-block:: python

    from spec2vec.logging_functions import add_logging_to_file

    add_logging_to_file("sample.log", loglevel="INFO")

If you want to write the logging messages to a local file while silencing the
stream of such messages, you can do the following:

.. code-block:: python

    from spec2vec.logging_functions import add_logging_to_file

    add_logging_to_file("sample.log", loglevel="INFO",
                        remove_stream_handlers=True)

"""
import logging
import logging.config
import sys
import matchms.logging_functions as matchms_logging


_formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(module)s:%(message)s')


def _init_logger(logger_name="spec2vec"):
    """Initialize spec2vec logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(_formatter)
    logger.addHandler(handler)
    logger.info('Completed configuring spec2vec logger.')


def set_spec2vec_logger_level(loglevel: str, logger_name="spec2vec"):
    """Update logging level to given loglevel.

    Parameters
    ----------
    loglevels
        Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    logger_name
        Default is "spec2vec". Change if logger name should be different.
    """
    matchms_logging.set_matchms_logger_level(loglevel=loglevel, logger_name=logger_name)


def add_logging_to_file(filename: str, loglevel: str = "INFO",
                        remove_stream_handlers: bool = False,
                        logger_name="spec2vec"):
    """Add logging to file.

    Current implementation does not change the initial logging stream,
    but simply adds a FileHandler to write logging entries to a file.

    Parameters
    ----------
    filename
        Name of file for write logging output.
    loglevels
        Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    remove_stream_handlers
        Set to True if only logging to file is desired.
    logger_name
        Default is "spec2vec". Change if logger name should be different.
    """
    matchms_logging.add_logging_to_file(filename=filename,
                                        loglevel=loglevel,
                                        remove_stream_handlers=remove_stream_handlers,
                                        logger_name=logger_name)


def reset_spec2vec_logger(logger_name="spec2vec"):
    """Reset spec2vec logger to initial state.

    This will remove all logging Handlers and initialize a new spec2vec logger.
    Use this function to reset previous changes made to the default spec2vec logger.

    Parameters
    ----------
    logger_name
        Default is "spec2vec". Change if logger name should be different.
    """
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    _init_logger()

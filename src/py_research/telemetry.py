# pylint: disable=C0415:import-outside-toplevel
# pylint: disable=E1136:unsubscriptable-object
# pylint: disable=W0622:redefined-builtin
"""Utilities for monitoring long-running functions via logging, tracing and metrics."""

import logging
import sys
from collections.abc import Iterable, Mapping
from functools import wraps
from logging import StreamHandler
from typing import IO, Literal

import structlog
from stqdm import stqdm
from tqdm import tqdm as base_tqdm
from tqdm.autonotebook import tqdm as atqdm


class TqdmHandler(StreamHandler):
    """A handler class which allows the cursor to stay on one line."""

    tqdm: base_tqdm | None = None

    def __init__(self, *args, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)

    def emit(self, record):  # noqa: D102
        try:
            msg = self.format(record)

            if self.tqdm is not None:
                self.tqdm.set_postfix_str(msg)
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)


def _check_streamlit():
    """Check whether python code is run within streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit


def tqdm(
    iterable: Iterable,
    desc: str | None = None,
    total: float | None = None,
    leave: bool | None = True,
    file: IO[str] | None = None,
    ncols: int | None = None,
    mininterval: float = 0.1,
    maxinterval: float = 10.0,
    miniters: float | None = None,
    ascii: bool | str | None = None,
    disable: bool | None = False,
    unit: str = "it",
    unit_scale: bool | float = False,
    dynamic_ncols: bool = False,
    smoothing: float = 0.3,
    bar_format: str | None = None,
    initial: float = 0,
    position: int | None = None,
    postfix: Mapping[str, object] | str | None = None,
    unit_divisor: float = 1000,
    write_bytes: bool = False,
    lock_args: tuple[bool | None, float | None] | tuple[bool | None] | None = None,
    nrows: int | None = None,
    colour: str | None = None,
    delay: float | None = 0,
    gui: bool = False,
    **other_kwargs,
) -> base_tqdm:
    """Return a tqdm instace adapted to the current environment.

    (Terminal, Jupyter or Streamlit)
    """
    kwargs = dict(
        iterable=iterable,
        desc=desc,
        total=total,
        leave=leave,
        file=file,
        ncols=ncols,
        mininterval=mininterval,
        maxinterval=maxinterval,
        miniters=miniters,
        ascii=ascii,
        disable=disable,
        unit=unit,
        unit_scale=unit_scale,
        dynamic_ncols=dynamic_ncols,
        smoothing=smoothing,
        bar_format=bar_format,
        initial=initial,
        position=position,
        postfix=postfix,
        unit_divisor=unit_divisor,
        write_bytes=write_bytes,
        lock_args=lock_args,
        nrows=nrows,
        colour=colour,
        delay=delay,
        gui=gui,
        **other_kwargs,
    )

    res_tqdm = stqdm(**kwargs) if _check_streamlit() else atqdm(**kwargs)  # type: ignore  # noqa: E501

    TqdmHandler.tqdm = res_tqdm

    return res_tqdm


def configure_logging(purpose: Literal["status", "report", "log"] = "status") -> None:
    """Auto-configure :py:mod:`structlog` based on logging purpose.

    Args:
        purpose:
            Which purpose the default global logger is supposed to fulfill.
            Can be any of:

            * ``status``:
                Inform about the current status of long-running functions
                via a single, changing console line.
            * ``report``:
                Record important steps / intermediate values / logical branches
                of large functions to the console.
            * ``log``:
                Produce a common log stream in JSON format.
    """
    structlog.configure(
        processors=[
            # If log level is too low, abort pipeline and throw away log entry.
            structlog.stdlib.filter_by_level,
            # Perform %-style formatting.
            structlog.stdlib.PositionalArgumentsFormatter(),
            *(
                [  # Add the name of the logger to event dict.
                    structlog.stdlib.add_logger_name,
                    # Add log level to event dict.
                    structlog.stdlib.add_log_level,
                    # Add a timestamp in ISO 8601 format.
                    structlog.processors.TimeStamper(fmt="iso"),
                    # If the "stack_info" key in the event dict is true, remove it and
                    # render the current stack trace in the "stack" key.
                    structlog.processors.StackInfoRenderer(),
                    # If the "exc_info" key in the event dict is either true or a
                    # sys.exc_info() tuple, remove "exc_info" and render the exception
                    # with traceback into the "exception" key.
                    structlog.processors.format_exc_info,
                    # If some value is in bytes, decode it to a unicode str.
                    structlog.processors.UnicodeDecoder(),
                    # Add callsite parameters.
                    structlog.processors.CallsiteParameterAdder(
                        {
                            structlog.processors.CallsiteParameter.FILENAME,
                            structlog.processors.CallsiteParameter.FUNC_NAME,
                            structlog.processors.CallsiteParameter.LINENO,
                        }
                    ),
                    structlog.dev.ConsoleRenderer()
                    if purpose != "log"
                    else structlog.processors.JSONRenderer(),
                ]
                if purpose != "status"
                else [structlog.dev.ConsoleRenderer(colors=False)]
            ),
        ],
        # `wrapper_class` is the bound logger that you get back from
        # get_logger(). This one imitates the API of `logging.Logger`.
        wrapper_class=structlog.stdlib.BoundLogger,
        # `logger_factory` is used to create wrapped loggers that are used for
        # OUTPUT. This one returns a `logging.Logger`. The final value (a JSON
        # string) from the final processor (`JSONRenderer`) will be passed to
        # the method of the same name as that you've called on the bound logger.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # Effectively freeze configuration after creating the first bound
        # logger.
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        handlers=[
            TqdmHandler(sys.stdout)
            if purpose == "status"
            else StreamHandler(sys.stdout)
        ],
        level=logging.DEBUG if purpose == "status" else logging.INFO,
        force=True,
    )


# Ensure that logging is configured to use `structlog.stdlib.BoundLogger`
# for `get_logger()`.
configure_logging()


@wraps(structlog.get_logger)
def get_logger(name: str | None = None, **kwds) -> structlog.stdlib.BoundLogger:
    """Typed interface for `structlog.get_logger`."""
    return structlog.get_logger(name, **kwds)

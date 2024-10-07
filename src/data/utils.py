# -*- coding: utf-8 -*-
import contextlib

import datasets


@contextlib.contextmanager
def progress_bar_disabled():
    pbar_enabled = datasets.utils.logging.is_progress_bar_enabled()
    try:
        if pbar_enabled:
            datasets.utils.logging.disable_progress_bar()
        yield
    finally:
        if pbar_enabled:
            datasets.utils.logging.enable_progress_bar()

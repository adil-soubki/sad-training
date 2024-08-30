# -*- coding: utf-8 -*
import os

from ..core.path import dirparent


PROMPT_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "prompts")


def load(*path: str) -> str:
    with open(os.path.join(PROMPT_DIR, *path), "r") as fd:
        return fd.read().strip()

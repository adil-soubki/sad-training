# -*- coding: utf-8 -*
import os

from .path import dirparent


KEY_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "keys")


def get(name: str) -> str:
    with open(os.path.join(KEY_DIR, name), "r") as fd:
        return fd.read().strip()

# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, TypeVar, Union
import os
import shelve


def safe_iter(arg: Union[Any, Iterable[Any]]) -> Iterable[Any]:
    """
    Makes something an iterable if it's just one item.

    Note:
        This does not consider strings to be an iterable since
        often you want a list of strings and not a list of characters.

    Args:
        arg: A single item or an iterable

    Returns:
        An iterable if it was a single item or the iterable.
    """
    if isinstance(arg, str) or not hasattr(arg, "__iter__"):
        return [arg]
    return arg


def save_iter(itr: Iterable[Any], path: str, mode: str = "w") -> None:
    """
    Writes an iterator line by line to the given file path.

    Args:
        itr (Iterator[Any]): An iterator to be stringified and saved.

    Examples:
        >>> german = language("german")
        >>> save_iter(
                map(
                    operator.itemgetter("english"),
                    itertools.islice(
                        filter(
                            german(contains("ja")),
                            sentences("german", "english")
                        ),
                        5
                    )
                ),
                "test.txt"
            )
        >>> with open("test.txt", "r") as fd:
                print(fd.read())
            After all, we have in you an expert who is in any case...
            To avoid expert-tourism under the cover of development...
            Mr Soulier, it is amusing that the version in which there...
            Miss McIntosh, we shall try to put this problem right in...
            We too have our drugs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, mode, encoding="utf-8") as fd:
        for first in itr:
            fd.write(str(first))
            for line in itr:
                fd.write("\n" + str(line))


class Filter:
    def __init__(self, fn: Callable[..., bool]):
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> bool:
        return self.fn(*args, **kwargs)

    def __and__(self, other: Filter) -> Filter:
        def fn(*args: Any, **kwargs: Any) -> bool:
            return self(*args, **kwargs) and other(*args, **kwargs)

        return Filter(fn)

    def __or__(self, other: Filter) -> Filter:
        def fn(*args: Any, **kwargs: Any) -> bool:
            return self(*args, **kwargs) or other(*args, **kwargs)

        return Filter(fn)

    def __invert__(self) -> Filter:
        return Filter(lambda *args, **kwargs: not self.fn(*args, **kwargs))


def shelf(path: str) -> Any:
    # Use an open handle if it exists.
    if hasattr(cache, path) and not hasattr(getattr(cache, path).dict, "closed"):
        return getattr(cache, path)
    # Otherwise open a new handle.
    setattr(cache, path, shelve.open(f"{path}.shelf"))
    return getattr(cache, path)


# TODO: Does this really need to support generator functions?
# TODO: Maybe this should be a class so it can have a "clear()".
# fmt: off
def cache(path: str) -> Callable[..., Any]:
    """
    A decorator that caches function calls to a shelf file.

    Args:
        path (str): A path to place/look for the shelf.

    Returns:
        A decorator that caches calls to the decorated function.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import pickle
            import inspect

            with shelf(path) as db:
                key = str(pickle.dumps(tuple(list(args) + list(kwargs.values()))))
                if inspect.isgeneratorfunction(fn):
                    if key not in db:
                        db[key] = list(fn(*args, **kwargs))
                    for item in db[key]:
                        yield item
                else:
                    if key not in db:
                        db[key] = fn(*args, **kwargs)
                    return db[key]
        return wrapper
    return decorator
# fmt: on

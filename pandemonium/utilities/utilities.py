import inspect
import sys
from functools import reduce
from typing import List, Type


def add_wrappers(base_env, wrappers: List[Type]):
    """Returns an environment wrapped with wrappers """
    return reduce(lambda e, wrapper: wrapper(e), wrappers, base_env)


def add_mixins(base_cls: Type, mixins: List[Type]):
    """Returns a new class with mixins added as bases """
    name = base_cls.__class__.__name__
    return reduce(lambda mixin, b: type(name, (mixin, b), {}), mixins, base_cls)


def get_all_classes(name, return_names: bool = True):
    members = inspect.getmembers(
        object=sys.modules[name],
        predicate=lambda member: inspect.isclass(
            member) and member.__module__ == name)
    if return_names:
        members = [m[0] for m in members]
    return members

import inspect
import sys


def get_all_classes(name, return_names: bool = True):
    members = inspect.getmembers(
        object=sys.modules[name],
        predicate=lambda member: inspect.isclass(
            member) and member.__module__ == name)
    if return_names:
        members = [m[0] for m in members]
    return members

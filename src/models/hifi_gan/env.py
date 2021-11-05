import os
import shutil
from typing import Any, Dict, Tuple


class AttrDict(Dict[Any, Any]):
    def __init__(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> None:
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

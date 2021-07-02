import json
from abc import ABC
import abc


class FileBase(ABC):
    @staticmethod
    @abc.abstractmethod  # 将方法标记为抽象的==>在子类中必须实现
    def load(inputs):
        """Retrieve data from the input source and return an object."""
        return

    @staticmethod
    @abc.abstractmethod
    def save(output, data):
        """Save the data object to the output."""
        return


class json_file(FileBase):

    @staticmethod
    def save(dict_config: dict, save_path: str):
        cfg = json.dumps(dict_config)
        with open(r'{}'.format(save_path) + '/cfg.json', 'w') as f:
            f.write(cfg)

    @staticmethod
    def load(file_path: str):
        cfg = json.load(open(file_path))
        return cfg

    @staticmethod
    def show(cfg: dict or str):
        if isinstance(cfg, dict):
            print(json.dumps(cfg, indent=4))
        elif isinstance(cfg, str) and cfg.split('.')[-1] == 'json':
            print(json.dumps(json_file.load(cfg), indent=2))
        else:
            raise ValueError



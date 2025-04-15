import yaml

class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        value = self[key]
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return ConfigDict(value)
        return value

    def to_dict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for key, value in other.items():
            self[key] = self._wrap(value)

def load_config(path):
    with open(path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    return ConfigDict(cfg_raw)

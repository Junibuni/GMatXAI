import yaml

class ConfigDict(dict):
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value
    
    def to_dict(self):
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(path):
    with open(path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    return ConfigDict(cfg_raw)

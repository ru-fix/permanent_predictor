import re
import copy

class AttrDict(dict):
    def __init__(self, mapping=None):
        super().__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    @staticmethod
    def __split(item):
        sub_items = re.findall(r'([^.]+)\.([^$]+)', item)
        return sub_items[0] if len(sub_items) == 1 else item

    @staticmethod
    def __single(item):
        return len(re.findall(r'\.', item)) == 0 if isinstance(item, str) else True

    def __to_attrdict(self, value):
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        return value

    def __setitem__(self, key, value):
        if isinstance(value, list):
            value = [self.__to_attrdict(item) for item in value]
        else:
            value = self.__to_attrdict(value)

        if not self.__single(key):
            top, others = self.__split(key)
            if top not in self.keys():
                super().__setitem__(top, AttrDict())
            self[top].__setitem__(others, value)
        else:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        if key in self.keys():
            return self.get(key)
        elif not self.__single(key):
            top, others = self.__split(key)
            return self.get(top)[others]
        else:
            return None

    __setattr__, __getattr__ = __setitem__, __getitem__

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

    def __update_recursive(self, old, new):
        for key, value in new.items():
            if isinstance(value, dict):
                old[key] = self.__update_recursive(old.get(key, {}), value)
            else:
                old[key] = value
        return old

    def update(self, *args, **kwargs):
        for arg in args:
            result = self.__update_recursive(self, arg)
            for key, value in result.items():
               self[key] = value
        result = self.__update_recursive(self, kwargs)
        for key, value in result.items():
            self[key] = value

    def __contains__(self, key):
        if key in self.keys():
            return True
        elif not isinstance(key,str):
            return False
        elif self.__single(key):
            return key in self.keys()
        else:
            top, others = self.__split(key)
            return others in self[top] if top in self.keys() else False

    def clone(self):
        return copy.deepcopy(self)
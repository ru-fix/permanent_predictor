class BaseGraphicalObject:
    def __init__(self, data, **kwargs):
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _preprocess_data(self):
        raise NotImplementedError('Not implement method \'preprocess_data\'')

    def _get_go_kwargs(self):
        return self.data

    def get(self):
        raise NotImplementedError('Not implement method \'get\'')


class BaseGraphicalObjectGroup:
    def __init__(self, data, **go_kwargs):
        self.data = data
        self.go_kwargs = go_kwargs

    def _recursive_get_depth(self, level_data, level=0):
        if isinstance(level_data, dict):
            iterator = iter(level_data)
            return self._recursive_get_depth(level_data[next(iterator)], level + 1)
        else:
            return level

    def get_depth(self):
        return self._recursive_get_depth(self.data)

    def _recursive_preprocess_data(self, data):
        raise NotImplementedError('Not implement method \'recursive_preprocess_data\'')

    def _preprocess_data(self):
        self.data = self._recursive_preprocess_data(self.data)

    def get(self):
        return self.data


class BaseBarGraphicalObject(BaseGraphicalObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

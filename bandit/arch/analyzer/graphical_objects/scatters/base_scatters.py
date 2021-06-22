from arch.analyzer.graphical_objects.base_go import BaseGraphicalObject


class BaseScatter(BaseGraphicalObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_scatter(kwargs):
        return dict(kwargs, type='scatter')

    def get(self):
        return self._get_scatter(self._get_go_kwargs())


class BaseMultiScatter(BaseScatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        return [self._get_scatter(kwargs) for kwargs in self._get_go_kwargs()]

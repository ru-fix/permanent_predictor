from arch.analyzer.graphical_objects.base_go import BaseBarGraphicalObject


class BaseBar(BaseBarGraphicalObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_bar(kwargs):
        return dict(kwargs, type='bar')

    def get(self):
        return self._get_bar(self._get_go_kwargs())


class BaseMultiBar(BaseBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        return [self._get_bar(kwargs) for kwargs in self._get_go_kwargs()]

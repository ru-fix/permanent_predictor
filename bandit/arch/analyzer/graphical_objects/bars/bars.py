from arch.analyzer.graphical_objects.bars.base_bars import *


class Bar(BaseBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preprocess_data()

    def _preprocess_data(self):
        self.data = dict(
            x=self.data[0],
            y=self.data[1]
        )


class MultiBar(BaseMultiBar):
    def __init__(self, *args, preprocess=True, **kwargs):
        super().__init__(*args, **kwargs)
        if preprocess:
            self._preprocess_data()

    def _preprocess_data(self):
        format_data = {}
        for dict_index, dict_data in enumerate(self.data):
            format_data[dict_index] = dict(
                # Arm index
                x=list(dict_data.keys()),
                # Percents
                y=[value[1] for value in dict_data.values()],
                # Counts
                text=[f"{value[0]}" for value in dict_data.values()],
                # Name
                name=f"{dict_index}"
            )

        self.data = list(format_data.values())


class PlayedAndOptimalPlaysBar(MultiBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

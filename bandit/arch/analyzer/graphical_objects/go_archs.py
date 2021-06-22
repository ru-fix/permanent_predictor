from arch.analyzer.graphical_objects.base_go import *
from arch.analyzer.graphical_objects.bars.bars import *
from arch.analyzer.graphical_objects.scatters.scatters import *

# Go's hierarchy
hierarchy = {
    # Scatters
    MultiScatterGroups: 0,
    MultiScatter: 1,
    Scatter: 2,
    # Bars
    MultiBar: 0,
    PlayedAndOptimalPlaysBar: 1,
    Bar: 2
}

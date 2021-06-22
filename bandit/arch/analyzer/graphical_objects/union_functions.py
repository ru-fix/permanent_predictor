from arch.analyzer.graphical_objects import go_archs


def union_scatter_with_scatter(first, second):
    return go_archs.MultiScatter([first.data, second.data], preprocess=False)


def union_multiscatter_with_scatter(first, second):
    new_multiscatter = go_archs.MultiScatter(first.data + [second.data], preprocess=False)
    return new_multiscatter


def union_multiscatter_with_multiscatter(first, second):
    data_group = {index: [scatter] for index, scatter in enumerate(first.data)}
    for index, scatter in enumerate(second.data):
        if index not in data_group:
            data_group[index] = []
        data_group[index].append(scatter)
    return go_archs.MultiScatterGroups(data_group, preprocess=False)


def union_multiscattergroup_with_muliscatter(first, second):
    assert first.get_depth() == 1, "MultiScatterGroup depth more than one"
    data_group = first.data
    for index, scatter in enumerate(second.data):
        if index not in data_group:
            data_group[index] = []
        data_group[index].append(scatter)
    return go_archs.MultiScatterGroups(data_group, preprocess=False)


def union_multigogroup_with_multigogroup(first, second):

    def update_data_group(data_group, source):
        for key, data in source.data.items():
            for index, go in enumerate(data):
                if index not in data_group:
                    data_group[index] = {}
                if key not in data_group[index]:
                    data_group[index][key] = []
                data_group[index][key].append(go)

    assert isinstance(first, second.__class__), f"Can not unite {first.__class__.__name} " \
                                                f"and {second.__class__.__name}"

    first_depth = first.get_depth()
    second_depth = second.get_depth()
    depth_difference = first_depth - second_depth
    if depth_difference < 0:
        first, second = second, first

    assert depth_difference <= 1, "MultiScatterGroups depth difference more than one"

    if second_depth == 1:

        if first_depth == 1:
            data_group = {}
            update_data_group(data_group, first)
        else:
            data_group = first.data

        update_data_group(data_group, second)

        return first.__class__(data_group, preprocess=False)

    else:
        raise ValueError(f"Unrealized union of {first.__class__.__name__} "
                         f"with depths {first_depth} and {second_depth}")


def union_playedandoptimalplaysbar_with_playedandoptimalplaysbar(first, second):
    return go_archs.MultiBar(first.data + [second.data[1]], preprocess=False)


def union_multibar_with_playedandoptimalplaysbar(first, second):
    return go_archs.MultiBar(first.data + [second.data[1]], preprocess=False)


def union(first, second):
    if go_archs.hierarchy[first.__class__] > go_archs.hierarchy[second.__class__]:
        first, second = second, first

    if isinstance(first, go_archs.BaseGraphicalObjectGroup) and isinstance(second, go_archs.BaseGraphicalObjectGroup):
        return union_multigogroup_with_multigogroup(first, second)

    if isinstance(first, go_archs.MultiScatterGroups):
        if isinstance(second, go_archs.MultiScatter):
            return union_multiscattergroup_with_muliscatter(first, second)

    elif isinstance(first, go_archs.MultiScatter):
        if isinstance(second, go_archs.Scatter):
            return union_multiscatter_with_scatter(first, second)
        elif isinstance(second, go_archs.MultiScatter):
            return union_multiscatter_with_multiscatter(first, second)
    elif isinstance(first, go_archs.Scatter):
        return union_scatter_with_scatter(first, second)

    if isinstance(first, go_archs.MultiBar):
        if isinstance(second, go_archs.PlayedAndOptimalPlaysBar):
            return union_multibar_with_playedandoptimalplaysbar(first, second)
        elif isinstance(second, go_archs.MultiBar):
            # union_multibar_and_multibar
            pass
        elif isinstance(second, go_archs.Bar):
            # union_multibar_with_bar
            pass
    elif isinstance(first, go_archs.PlayedAndOptimalPlaysBar):
        if isinstance(second, go_archs.PlayedAndOptimalPlaysBar):
            return union_playedandoptimalplaysbar_with_playedandoptimalplaysbar(first, second)
    elif isinstance(first, go_archs.Bar):
        # union_bar_with_bar
        pass

    return TypeError(f"Can't union {first.__class__.__name__} with {second.__class__.__name__}")
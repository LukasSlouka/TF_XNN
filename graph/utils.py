import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Creates colormap with shifted midpoint (useful for speedup
    where 1.0 is midpoint instead of 0)

    @param cmap : The matplotlib colormap to be altered
    @param start : Offset from lowest point in the colormap's range. Defaults to 0.0 (no lower ofset). Should be between 0.0 and `midpoint`.
    @param midpoint : The new center of the colormap. Defaults to 0.5 (no shift). Should be between 0.0 and 1.0. In general, this should
                      be  1 - vmax/(vmax + abs(vmin)) For example if your data range from -15.0 to +5.0 and you want the center of the
                      colormap at 0.0, `midpoint` should be set to  1 - 5/(5 + 15)) or 0.75
    @param stop : Offset from highets point in the colormap's range. Defaults to 1.0 (no upper ofset). Should be between `midpoint` and 1.0.

    Credit goes to: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def get_cmap(midpoint: float):
    """
    Gets Red White Green colormap with 255 colors
    :param midpoint: midpoint offset
    :return: colormap
    """
    cmap = LinearSegmentedColormap.from_list(
        name='speedup_cmap',
        colors=['red', 'white', 'green'],
        N=255
    )
    return shiftedColorMap(cmap, midpoint=midpoint, name='shifted')


def get_position_of_one(min: float, max: float) -> float:
    """
    Maps 1.0 value in range min-max to range 0-1
    :param min: minimum
    :param max: maximum
    :return:
    """
    mm = max - min
    return (1.0 - min) / mm

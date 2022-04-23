import seaborn as sns
import matplotlib.pyplot as plt


def set_size(width, fraction = 1, subplots = (1, 1), scale_height = 1, width_inch = None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    scale_height : float
            Increase proportion in height
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width_inch is None:
        if width == 'thesis':
            width_pt = 426.79135
        elif width == 'beamer':
            width_pt = 307.28987
        elif width == "article":
            width_pt = 430.00462
        else:
            width_pt = width
        
        # Width of figure (in pts)
        fig_width_pt = width_pt * fraction
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
    
    else:
        fig_width_in = width_inch



    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2


    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * scale_height

    return (fig_width_in, fig_height_in)


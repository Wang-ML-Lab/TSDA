import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def arc_patch(center, radius, theta1, theta2, ax=None, resolution=50, **kwargs):
    # be sure that theta is in radian!!
    # make sure ax is not empty
    if ax is None:
        ax = plt.gca()
    # generate the points
    theta = np.linspace(theta1, theta2, resolution)
    points = np.vstack((radius*np.cos(theta) + center[0], 
                        radius*np.sin(theta) + center[1]))
    # build the polygon and add it to the axes
    # print(**kwargs)
    poly = mpatches.Polygon(points.T, closed=True, **kwargs)
    # poly = mpatches.Polygon(points.T, closed=True, fill=True, color='blue')
    ax.add_patch(poly)
    ax.plot()
    return poly


def domain_draw(ax, center, data, label, scale, ms):
    # doing normalization
    data_mean = data.mean(0, keepdims=True)
    data_std = data.std(0, keepdims=True)
    data = (data - data_mean) / data_std
    
    plot_data = data * scale
    # be careful about broadcast!!
    plot_data += center

    # just 2 classes
    for i in range(2):
        data_sub = plot_data[label == i, :]
        ax.plot(data_sub[:, 0], data_sub[:, 1], ['x', '.'][i], color=['r', 'b'][i], alpha=0.5,
            ms=ms
        )

def line_draw(ax, center, angle, radius):

    # add half line to identify the red point
    shift2 = 1.5 * radius * np.array([-np.sin(angle), np.cos(angle)])
    start_xy = center
    end_xy = center + shift2
    (line_x, line_y) = zip(*[start_xy, end_xy])
    ax.add_line(Line2D(line_x, line_y, linewidth=2, color='black'))
    ax.plot()

    # the true boundary
    shift = radius * np.array([np.cos(angle), np.sin(angle)])
    start_xy = center - shift
    end_xy = center + shift

    (line_x, line_y) = zip(*[start_xy, end_xy])

    ax.add_line(Line2D(line_x, line_y, linewidth=2, color='black'))
    ax.plot()



# 开始写将数据微缩plot出来的代码，在这里，同时还要画根不长的分界线

"""
**********
Matplotlib
**********

Draw networks with matplotlib.

See Also
--------

matplotlib:     http://matplotlib.org/

pygraphviz:     http://pygraphviz.github.io/

"""
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
    shell_layout,
    circular_layout,
    kamada_kawai_layout,
    spectral_layout,
    spring_layout,
    random_layout,
    planar_layout,
)

__all__ = [
    "draw",
    "draw_networkx",
    "draw_networkx_nodes",
    "draw_networkx_edges",
    "draw_networkx_labels",
    "draw_networkx_edge_labels",
    "draw_circular",
    "draw_kamada_kawai",
    "draw_random",
    "draw_spectral",
    "draw_spring",
    "draw_planar",
    "draw_shell",
]


def draw(G, pos=None, ax=None, **kwds):
    """Draw the graph G with Matplotlib.

    Draw the graph as a simple representation with no node
    labels or edge labels and using the full Matplotlib figure area
    and no axis labels by default.  See draw_networkx() for more
    full-featured drawing that allows title, axis labels etc.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See :py:mod:`networkx.drawing.layout` for functions that
       compute node positions.

    ax : Matplotlib Axes object, optional
       Draw the graph in specified Matplotlib axes.

    kwds : optional keywords
       See networkx.draw_networkx() for a description of optional keywords.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout

    See Also
    --------
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()

    Notes
    -----
    This function has the same name as pylab.draw and pyplot.draw
    so beware when using `from networkx import *`

    since you might overwrite the pylab.draw function.

    With pyplot use

    >>> import matplotlib.pyplot as plt
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)  # networkx draw()
    >>> plt.draw()  # pyplot draw()

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        cf = plt.gcf()
    else:
        cf = ax.get_figure()
    cf.set_facecolor("w")
    if ax is None:
        if cf._axstack() is None:
            ax = cf.add_axes((0, 0, 1, 1))
        else:
            ax = cf.gca()

    if "with_labels" not in kwds:
        kwds["with_labels"] = "labels" in kwds

    draw_networkx(G, pos=pos, ax=ax, **kwds)
    ax.set_axis_off()
    plt.draw_if_interactive()
    return



def draw_networkx(G, pos=None, arrows=True, acc_draw=False, with_labels=True, with_boundary=True, **kwds):
    """Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See :py:mod:`networkx.drawing.layout` for functions that
       compute node positions.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
        For directed graphs, choose the style of the arrowsheads.
        See :py:class: `matplotlib.patches.ArrowStyle` for more
        options.

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    with_labels :  bool, optional (default=True)
       Set to True to draw labels on the nodes.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist : list, optional (default G.nodes())
       Draw only specified nodes

    edgelist : list, optional (default=G.edges())
       Draw only specified edges

    node_size : scalar or array, optional (default=300)
       Size of nodes.  If an array is specified it must be the
       same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
       Node color. Can be a single color or a sequence of colors with the same
       length as nodelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the cmap and vmin,vmax parameters. See
       matplotlib.scatter for more details.

    node_shape :  string, optional (default='o')
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8'.

    alpha : float, optional (default=None)
       The node and edge transparency

    cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of nodes

    vmin,vmax : float, optional (default=None)
       Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)

    width : float, optional (default=1.0)
       Line width of edges

    edge_color : color or array of colors (default='k')
       Edge color. Can be a single color or a sequence of colors with the same
       length as edgelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_cmap : Matplotlib colormap, optional (default=None)
       Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional (default=None)
       Minimum and maximum for edge colormap scaling

    style : string, optional (default='solid')
       Edge line style (solid|dashed|dotted,dashdot)

    labels : dictionary, optional (default=None)
       Node labels in a dictionary keyed by node of text labels

    font_size : int, optional (default=12)
       Font size for text labels

    font_color : string, optional (default='k' black)
       Font color string

    font_weight : string, optional (default='normal')
       Font weight

    font_family : string, optional (default='sans-serif')
       Font family

    label : string, optional
       Label for graph legend

    kwds : optional keywords
       See networkx.draw_networkx_nodes(), networkx.draw_networkx_edges(), and
       networkx.draw_networkx_labels() for a description of optional keywords.

    Notes
    -----
    For directed graphs, arrows  are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout

    >>> import matplotlib.pyplot as plt
    >>> limits = plt.axis("off")  # turn of axis

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    valid_node_kwds = (
        # some of my own parameters:
        "node_angles",
        "node_radius",
        "data_points",
        "data_label",
        "data_domain",
        "node_acc",

        "nodelist",
        "node_size",
        "node_color",
        "node_shape",
        "alpha",
        "cmap",
        "vmin",
        "vmax",
        "ax",
        "linewidths",
        "edgecolors",
        "label",
    )

    valid_edge_kwds = (
        "edgelist",
        "width",
        "edge_color",
        "style",
        "alpha",
        "arrowstyle",
        "arrowsize",
        "edge_cmap",
        "edge_vmin",
        "edge_vmax",
        "ax",
        "label",
        "node_size",
        "nodelist",
        "node_shape",
        "connectionstyle",
        "min_source_margin",
        "min_target_margin",
    )

    valid_label_kwds = (
        "labels",
        "font_size",
        "font_color",
        "font_family",
        "font_weight",
        "alpha",
        "bbox",
        "ax",
        "horizontalalignment",
        "verticalalignment",
    )

    valid_kwds = valid_node_kwds + valid_edge_kwds + valid_label_kwds

    if any([k not in valid_kwds for k in kwds]):
        invalid_args = ", ".join([k for k in kwds if k not in valid_kwds])
        raise ValueError(f"Received invalid argument(s): {invalid_args}")

    node_kwds = {k: v for k, v in kwds.items() if k in valid_node_kwds}
    edge_kwds = {k: v for k, v in kwds.items() if k in valid_edge_kwds}
    label_kwds = {k: v for k, v in kwds.items() if k in valid_label_kwds}

    if pos is None:
        pos = nx.drawing.spring_layout(G)  # default to spring layout

    # draw_networkx_nodes(G, pos, **node_kwds)

    # use my own nodes instead
    if acc_draw:
        my_draw_networkx_nodes_acc(G, pos, **node_kwds)
    else:
        my_draw_networkx_nodes(G, pos, with_boundary=with_boundary, **node_kwds)

    draw_networkx_edges(G, pos, arrows=arrows, **edge_kwds)
    if with_labels:
        draw_networkx_labels(G, pos, **label_kwds)
    plt.draw_if_interactive()


# this time I will draw the predict results
def my_draw_networkx_nodes(
    G,
    pos,
    node_angles=None,
    node_radius=1,
    nodelist=None,
    ax=None,
    label=None,
    data_points=None,
    data_label=None,
    data_domain=None,
    with_boundary=True,
    # node_acc=None,
    **kwds,
):
    from collections.abc import Iterable

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)
    
    # xys = np.asarray([pos[v] for v in nodelist])
    # num_node = 
    # for i in range(num_node):
    for v in nodelist:
        plot_index = data_domain == v
        sub_data = data_points[plot_index, :]
        # print(sub_data.mean())
        sub_label = data_label[plot_index]
        # arc_patch(pos[v], node_radius, node_angles[v], node_angles[v] + np.pi, ax, fill=True, color='blue')
        # arc_patch(pos[v], node_radius, node_angles[v]+np.pi, node_angles[v] + 2 * np.pi, ax, fill=True, color='red')
        # domain_draw(ax, pos[v], sub_data, sub_label, scale=0.031, ms=2)
        domain_draw(ax, pos[v], sub_data, sub_label, scale=node_radius*0.45, ms=2)
        if with_boundary:
            line_draw(ax, pos[v], node_angles[v], radius=node_radius)
 
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    

def my_draw_networkx_nodes_acc(
    G,
    pos,
    nodelist=None,
    node_radius=1,
    # node_angles=None,
    alpha=None,
    cmap=matplotlib.cm.get_cmap('rainbow'),
    vmin=None,
    vmax=None,
    ax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
    node_acc=None,
    **kwds,
):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist : list, optional
       Draw only specified nodes (default G.nodes())

    node_size : scalar or array
       Size of nodes (default=300).  If an array is specified it must be the
       same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
       Node color. Can be a single color or a sequence of colors with the same
       length as nodelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the cmap and vmin,vmax parameters. See
       matplotlib.scatter for more details.

    node_shape :  string
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8' (default='o').

    alpha : float or array of floats
       The node transparency.  This can be a single alpha value (default=None),
       in which case it will be applied to all the nodes of color. Otherwise,
       if it is an array, the elements of alpha will be applied to the colors
       in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None)

    vmin,vmax : floats
       Minimum and maximum for node colormap scaling (default=None)

    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)

    edgecolors : [None | scalar | sequence]
       Colors of node borders (default = node_color)

    label : [None| string]
       Label for legend

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nodes = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    from collections.abc import Iterable

    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if len(nodelist) == 0:  # empty nodelist, no drawing
        return PathCollection(None)

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise nx.NetworkXError(f"Node {e} has no position.") from e
    except ValueError as e:
        raise nx.NetworkXError("Bad value in node positions.") from e

    # if isinstance(alpha, Iterable):
    #     node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
    #     alpha = None

    # cmap = matplotlib.cm.get_cmap('rainbow')
    # l_color = [ cmap(ele)[:3] 
    for v in nodelist:
        l_color = cmap(node_acc[v] / 100)[:3] 
        arc_patch(pos[v], node_radius, 0, 2 * np.pi, ax, resolution=50, fill=True, color=l_color)

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )




def draw_networkx_nodes(
    G,
    pos,
    nodelist=None,
    node_size=300,
    node_color="#1f78b4",
    node_shape="o",
    alpha=None,
    cmap=None,
    vmin=None,
    vmax=None,
    ax=None,
    linewidths=None,
    edgecolors=None,
    label=None,
):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist : list, optional
       Draw only specified nodes (default G.nodes())

    node_size : scalar or array
       Size of nodes (default=300).  If an array is specified it must be the
       same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
       Node color. Can be a single color or a sequence of colors with the same
       length as nodelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the cmap and vmin,vmax parameters. See
       matplotlib.scatter for more details.

    node_shape :  string
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8' (default='o').

    alpha : float or array of floats
       The node transparency.  This can be a single alpha value (default=None),
       in which case it will be applied to all the nodes of color. Otherwise,
       if it is an array, the elements of alpha will be applied to the colors
       in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None)

    vmin,vmax : floats
       Minimum and maximum for node colormap scaling (default=None)

    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)

    edgecolors : [None | scalar | sequence]
       Colors of node borders (default = node_color)

    label : [None| string]
       Label for legend

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nodes = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    from collections.abc import Iterable

    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PathCollection
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = list(G)

    if len(nodelist) == 0:  # empty nodelist, no drawing
        return PathCollection(None)

    try:
        xy = np.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise nx.NetworkXError(f"Node {e} has no position.") from e
    except ValueError as e:
        raise nx.NetworkXError("Bad value in node positions.") from e

    if isinstance(alpha, Iterable):
        node_color = apply_alpha(node_color, alpha, nodelist, cmap, vmin, vmax)
        alpha = None

    node_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        marker=node_shape,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        label=label,
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    node_collection.set_zorder(2)
    return node_collection



def draw_networkx_edges(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle="-|>",
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=True,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    connectionstyle=None,
    min_source_margin=0,
    min_target_margin=0,
):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    edgelist : collection of edge tuples
       Draw only specified edges(default=G.edges())

    width : float, or array of floats
       Line width of edges (default=1.0)

    edge_color : color or array of colors (default='k')
       Edge color. Can be a single color or a sequence of colors with the same
       length as edgelist. Color can be string, or rgb (or rgba) tuple of
       floats from 0-1. If numeric values are specified they will be
       mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    alpha : float
       The edge transparency (default=None)

    edge_ cmap : Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax : floats
       Minimum and maximum for edge colormap scaling (default=None)

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    arrows : bool, optional (default=True)
       For directed graphs, if True draw arrowheads.
       Note: Arrows will be the same color as edges.

    arrowstyle : str, optional (default='-|>')
       For directed graphs, choose the style of the arrow heads.
       See :py:class: `matplotlib.patches.ArrowStyle` for more
       options.

    arrowsize : int, optional (default=10)
       For directed graphs, choose the size of the arrow head head's length and
       width. See :py:class: `matplotlib.patches.FancyArrowPatch` for attribute
       `mutation_scale` for more info.

    connectionstyle : str, optional (default=None)
       Pass the connectionstyle parameter to create curved arc of rounding
       radius rad. For example, connectionstyle='arc3,rad=0.2'.
       See :py:class: `matplotlib.patches.ConnectionStyle` and
       :py:class: `matplotlib.patches.FancyArrowPatch` for more info.

    label : [None| string]
       Label for legend

    min_source_margin : int, optional (default=0)
       The minimum margin (gap) at the begining of the edge at the source.

    min_target_margin : int, optional (default=0)
       The minimum margin (gap) at the end of the edge at the target.

    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges

    list of matplotlib.patches.FancyArrowPatch
        `FancyArrowPatch` instances of the directed edges

    Depending whether the drawing includes arrows or not.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False. Be sure to include `node_size` as a
    keyword argument; arrows are drawn considering the size of nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import colorConverter, Colormap, Normalize
        from matplotlib.collections import LineCollection
        from matplotlib.patches import FancyArrowPatch
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if len(edgelist) == 0:  # no edges!
        if not G.is_directed() or not arrows:
            return LineCollection(None)
        else:
            return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.alltrue([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    if not G.is_directed() or not arrows:
        edge_collection = LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            transOffset=ax.transData,
            alpha=alpha,
        )

        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)

        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)

        return edge_collection

    arrow_collection = None

    if G.is_directed() and arrows:
        # Note: Waiting for someone to implement arrow to intersection with
        # marker.  Meanwhile, this works well for polygons with more than 4
        # sides and circle.

        def to_marker_edge(marker_size, marker):
            if marker in "s^>v<d":  # `large` markers need extra space
                return np.sqrt(2 * marker_size) / 2
            else:
                return np.sqrt(marker_size) / 2

        # Draw arrows with `matplotlib.patches.FancyarrowPatch`
        arrow_collection = []
        mutation_scale = arrowsize  # scale factor of arrow head

        # FancyArrowPatch doesn't handle color strings
        arrow_colors = colorConverter.to_rgba_array(edge_color, alpha)
        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            shrink_source = 0  # space from source to tail
            shrink_target = 0  # space from  head to target
            if np.iterable(node_size):  # many node sizes
                source, target = edgelist[i][:2]
                source_node_size = node_size[nodelist.index(source)]
                target_node_size = node_size[nodelist.index(target)]
                shrink_source = to_marker_edge(source_node_size, node_shape)
                shrink_target = to_marker_edge(target_node_size, node_shape)
            else:
                shrink_source = shrink_target = to_marker_edge(node_size, node_shape)

            if shrink_source < min_source_margin:
                shrink_source = min_source_margin

            if shrink_target < min_target_margin:
                shrink_target = min_target_margin

            if len(arrow_colors) == len(edge_pos):
                arrow_color = arrow_colors[i]
            elif len(arrow_colors) == 1:
                arrow_color = arrow_colors[0]
            else:  # Cycle through colors
                arrow_color = arrow_colors[i % len(arrow_colors)]

            if np.iterable(width):
                if len(width) == len(edge_pos):
                    line_width = width[i]
                else:
                    line_width = width[i % len(width)]
            else:
                line_width = width

            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=arrowstyle,
                shrinkA=shrink_source,
                shrinkB=shrink_target,
                mutation_scale=mutation_scale,
                color=arrow_color,
                linewidth=line_width,
                connectionstyle=connectionstyle,
                linestyle=style,
                zorder=1,
            )  # arrows go behind nodes

            # There seems to be a bug in matplotlib to make collections of
            # FancyArrowPatch instances. Until fixed, the patches are added
            # individually to the axes instance.
            arrow_collection.append(arrow)
            ax.add_patch(arrow)

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return arrow_collection



def draw_networkx_labels(
    G,
    pos,
    labels=None,
    font_size=12,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
):
    """Draw node labels on the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    labels : dictionary, optional (default=None)
       Node labels in a dictionary keyed by node of text labels
       Node-keys in labels should appear as keys in `pos`.
       If needed use: `{n:lab for n,lab in labels.items() if n in pos}`

    font_size : int
       Font size for text labels (default=12)

    font_color : string
       Font color string (default='k' black)

    font_family : string
       Font family (default='sans-serif')

    font_weight : string
       Font weight (default='normal')

    alpha : float or None
       The text transparency (default=None)

    horizontalalignment : {'center', 'right', 'left'}
       Horizontal alignment (default='center')

    verticalalignment : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
        Vertical alignment (default='center')

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.


    Returns
    -------
    dict
        `dict` of labels keyed on the nodes

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = {n: n for n in G.nodes()}

    text_items = {}  # there is no text collection so we'll fake one
    for n, label in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transData,
            bbox=bbox,
            clip_on=True,
        )
        text_items[n] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items



def draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    alpha : float or None
       The text transparency (default=None)

    edge_labels : dictionary
       Edge labels in a dictionary keyed by edge two-tuple of text
       labels (default=None). Only labels for the keys in the dictionary
       are drawn.

    label_pos : float
       Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int
       Font size for text labels (default=12)

    font_color : string
       Font color string (default='k' black)

    font_weight : string
       Font weight (default='normal')

    font_family : string
       Font family (default='sans-serif')

    bbox : Matplotlib bbox
       Specify text box shape and colors.

    clip_on : bool
       Turn on clipping at axis boundaries (default=True)

    horizontalalignment : {'center', 'right', 'left'}
       Horizontal alignment (default='center')

    verticalalignment : {'center', 'top', 'bottom', 'baseline', 'center_baseline'}
        Vertical alignment (default='center')

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    Returns
    -------
    dict
        `dict` of labels keyed on the edges

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=True,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items



def draw_circular(G, **kwargs):
    """Draw the graph G with a circular layout.

    Parameters
    ----------
    G : graph
       A networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    draw(G, circular_layout(G), **kwargs)



def draw_kamada_kawai(G, **kwargs):
    """Draw the graph G with a Kamada-Kawai force-directed layout.

    Parameters
    ----------
    G : graph
       A networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    draw(G, kamada_kawai_layout(G), **kwargs)



def draw_random(G, **kwargs):
    """Draw the graph G with a random layout.

    Parameters
    ----------
    G : graph
       A networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    draw(G, random_layout(G), **kwargs)



def draw_spectral(G, **kwargs):
    """Draw the graph G with a spectral 2D layout.

    Using the unnormalized Laplacian, the layout shows possible clusters of
    nodes which are an approximation of the ratio cut. The positions are the
    entries of the second and third eigenvectors corresponding to the
    ascending eigenvalues starting from the second one.

    Parameters
    ----------
    G : graph
       A networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    draw(G, spectral_layout(G), **kwargs)



def draw_spring(G, **kwargs):
    """Draw the graph G with a spring layout.

    Parameters
    ----------
    G : graph
       A networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    draw(G, spring_layout(G), **kwargs)



def draw_shell(G, **kwargs):
    """Draw networkx graph with shell layout.

    Parameters
    ----------
    G : graph
       A networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    nlist = kwargs.get("nlist", None)
    if nlist is not None:
        del kwargs["nlist"]
    draw(G, shell_layout(G, nlist=nlist), **kwargs)



def draw_planar(G, **kwargs):
    """Draw a planar networkx graph with planar layout.

    Parameters
    ----------
    G : graph
       A planar networkx graph

    kwargs : optional keywords
       See networkx.draw_networkx() for a description of optional keywords,
       with the exception of the pos parameter which is not used by this
       function.
    """
    draw(G, planar_layout(G), **kwargs)



def apply_alpha(colors, alpha, elem_list, cmap=None, vmin=None, vmax=None):
    """Apply an alpha (or list of alphas) to the colors provided.

    Parameters
    ----------

    colors : color string, or array of floats
       Color of element. Can be a single color format string (default='r'),
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    alpha : float or array of floats
       Alpha values for elements. This can be a single alpha value, in
       which case it will be applied to all the elements of color. Otherwise,
       if it is an array, the elements of alpha will be applied to the colors
       in order (cycling through alpha multiple times if necessary).

    elem_list : array of networkx objects
       The list of elements which are being colored. These could be nodes,
       edges or labels.

    cmap : matplotlib colormap
       Color map for use if colors is a list of floats corresponding to points
       on a color mapping.

    vmin, vmax : float
       Minimum and maximum values for normalizing colors if a color mapping is
       used.

    Returns
    -------

    rgba_colors : numpy ndarray
        Array containing RGBA format values for each of the node colours.

    """
    from itertools import islice, cycle

    try:
        import numpy as np
        from matplotlib.colors import colorConverter
        import matplotlib.cm as cm
    except ImportError as e:
        raise ImportError("Matplotlib required for draw()") from e

    # If we have been provided with a list of numbers as long as elem_list,
    # apply the color mapping.
    if len(colors) == len(elem_list) and isinstance(colors[0], Number):
        mapper = cm.ScalarMappable(cmap=cmap)
        mapper.set_clim(vmin, vmax)
        rgba_colors = mapper.to_rgba(colors)
    # Otherwise, convert colors to matplotlib's RGB using the colorConverter
    # object.  These are converted to numpy ndarrays to be consistent with the
    # to_rgba method of ScalarMappable.
    else:
        try:
            rgba_colors = np.array([colorConverter.to_rgba(colors)])
        except ValueError:
            rgba_colors = np.array([colorConverter.to_rgba(color) for color in colors])
    # Set the final column of the rgba_colors to have the relevant alpha values
    try:
        # If alpha is longer than the number of colors, resize to the number of
        # elements.  Also, if rgba_colors.size (the number of elements of
        # rgba_colors) is the same as the number of elements, resize the array,
        # to avoid it being interpreted as a colormap by scatter()
        if len(alpha) > len(rgba_colors) or rgba_colors.size == len(elem_list):
            rgba_colors = np.resize(rgba_colors, (len(elem_list), 4))
            rgba_colors[1:, 0] = rgba_colors[0, 0]
            rgba_colors[1:, 1] = rgba_colors[0, 1]
            rgba_colors[1:, 2] = rgba_colors[0, 2]
        rgba_colors[:, 3] = list(islice(cycle(alpha), len(rgba_colors)))
    except TypeError:
        rgba_colors[:, -1] = alpha
    return rgba_colors
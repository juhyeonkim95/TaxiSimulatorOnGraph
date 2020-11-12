import osmnx as ox
import networkx as nx
import numpy as np
import copy
from shapely.geometry import LineString, MultiLineString
from shapely import ops


def generate_square_multidi_graph(width, height, mean_distance=300, std=100, min_distance=100) ->nx.MultiDiGraph:
    '''
    Generate graph for simple grid case.
    :param width: number of columns
    :param height: number of rows
    :param mean_distance: mean distance of the road
    :param std: std of the road
    :param min_distance: minimum distance of the road
    :return: networkx Graph object
    '''
    G = nx.MultiDiGraph()
    N = width * height
    G.add_nodes_from(list(range(0, N)))

    lengths = []
    ebunchs = []
    for i in range(width):
        for j in range(height):
            a = j * width + i
            b = a + 1
            c = a + width
            # create roads on right direction.
            if i < width - 1:
                l_right = np.random.normal(mean_distance, std)
                l_right = max(min_distance, l_right)
                ebunchs.extend([(a, b, {'length': l_right,'u':a, 'v':b}),
                                (b, a, {'length': l_right,'v':b, 'u':a})])
                lengths.extend([l_right, l_right])
            # create roads on bottom direction.
            if j < height - 1:
                l_down = np.random.normal(mean_distance, std)
                l_down = max(min_distance, l_down)
                ebunchs.extend([(a, c, {'length': l_down, 'u': a, 'v': c}),
                                (c, a, {'length': l_down, 'v': c, 'u': a})])

                lengths.extend([l_down, l_down])

    G.add_edges_from(ebunchs)
    return G


def check_highway_type_is_to_be_removed(highway_type, remove_list=None):
    '''
    Checks whether highway type is to be removed type.
    :param highway_type: This can be both string of list of strings.
    :param remove_list: A list of to be removed types. For details, check https://wiki.openstreetmap.org/wiki/Key:highway.
    :return: True for remove false for not remove.
    '''
    if remove_list is None:
        remove_list = ['residential',
                      'living_street',
                      'rest_area',
                      #'trunk',
                      #'motorway',
                      #'motorway_link',
                      #'trunk_link',
                      #'primary_link',
                      #'secondary_link',
                      #'tertiary_link',
                      'road',
                      'bus_guideway',
                      'disused',
                      'sidewalk',
                      'crossing'
                      ]
    if type(highway_type) == list:
        for h in highway_type:
            if h in remove_list:
                return True
        return False
    return highway_type in remove_list


def get_all_types(G: nx.MultiDiGraph):
    '''
    :param G: Graph
    :return: All 'highway' types in the road network.
    '''
    highway_type_set = set()
    for e in G.edges(data=True):
        u, v, info = e
        highway_type = info['highway']
        if type(highway_type) == list:
            for h in highway_type:
                highway_type_set.add(h)
        else:
            highway_type_set.add(highway_type)

    return highway_type_set


def print_graph_info(G):
    print("Number of nodes: %d, Number of edges: %d" % (G.number_of_nodes(), G.number_of_edges()))


# Functions for graph simplification
def simplify_graph_remove_unimportant_roads(G_original: nx.MultiDiGraph):
    '''
    Removes all unimportant roads.
    :param G_original: networkx graph object.
    :return: SImplified networkx graph object.
    '''
    G = G_original.copy()
    to_remove = []
    for e in list(G.edges(data=True, keys=True)):
        u, v, i, info = e
        highway_type = info['highway']
        if check_highway_type_is_to_be_removed(highway_type) and info['length'] < 1000:
            to_remove.append((u, v, i))

    G.remove_edges_from(to_remove)
    G = ox.remove_isolated_nodes(G)
    # print(get_all_types(G))

    G_component = ox.get_largest_component(G, strongly=True)

    print_graph_info(G_original)
    print_graph_info(G)
    print_graph_info(G_component)

    return G_component


def simplify_graph_remove_boundary_nodes(G_original: nx.MultiDiGraph):
    '''
    Removes dangling roads at the boundary.
    :param G_original: networkx graph object.
    :return: SImplified networkx graph object.
    '''
    G = G_original.copy()
    while True:
        to_remove = []
        for n, info in list(G.nodes(data=True)):
            ins = list(G.predecessors(n))
            outs = list(G.successors(n))

            if G.in_degree(n) == 1 and G.out_degree(n) == 1 and len(ins) == 1 and len(outs) == 1 and ins[0] == outs[0]:
                to_remove.append(n)
        if len(to_remove) == 0:
            break
        G.remove_nodes_from(to_remove)

    print("Boundary Removed")
    print_graph_info(G)
    return G


def simplify_graph_remove11(G_original: nx.MultiDiGraph):
    '''
    Simplifies ->-> shaped road to ->.
    :param G_original: networkx graph object.
    :return: SImplified networkx graph object.
    '''
    G = G_original.copy()

    def set_geometry(e):
        u, v, _, edge_info = e
        if 'geometry' not in edge_info:
            ux = G.nodes[u]['x']
            uy = G.nodes[u]['y']
            vx = G.nodes[v]['x']
            vy = G.nodes[v]['y']
            edge_info['geometry'] = LineString([[ux, uy], [vx, vy]])

    while True:
        to_remove = []
        for n, info in list(G.nodes(data=True)):
            # a -> b -> c

            outs = list(G.out_edges(n, data=True, keys=True))
            ins = list(G.in_edges(n, data=True, keys=True))
            if len(ins) == 1 and len(outs) == 1:  # G.in_degree(n) == 2 and G.out_degree(n) == 2 and len(ins) ==2 and len(outs) == 2:
                a = ins[0][0]
                b = n
                c = outs[0][1]
                if a!=c and b!=c and a!=b:

                    for e in (ins + outs):
                        set_geometry(e)

                    l_ab = ins[0][3]
                    l_bc = outs[0][3]

                    l_ac = copy.deepcopy(l_ab)
                    gac = MultiLineString([l_ab['geometry'], l_bc['geometry']])
                    gac = ops.linemerge(gac)

                    l_ac['length'] = l_ab['length'] + l_bc['length']
                    l_ac['geometry'] = gac

                    G.add_edge(a, c, **l_ac)
                    G.remove_node(b)

                    to_remove.append(b)

        if len(to_remove) == 0:
            break

    print("-- Removed")
    print_graph_info(G)
    return G


def simplify_graph_remove22(G_original: nx.MultiDiGraph):
    '''
    Simplifies <=><=> shaped road to <=>.
    :param G_original: networkx graph object.
    :return: SImplified networkx graph object.
    '''
    G = G_original.copy()

    def set_geometry(e):
        u, v, _, edge_info = e
        if 'geometry' not in edge_info:
            ux = G.nodes[u]['x']
            uy = G.nodes[u]['y']
            vx = G.nodes[v]['x']
            vy = G.nodes[v]['y']
            edge_info['geometry'] = LineString([[ux, uy], [vx, vy]])

    while True:
        to_remove = []
        for n in list(G.nodes()):
            # a0 -> b0 -> c0
            # a1 <- b1 <- c1

            outs = list(G.out_edges(n, data=True, keys=True))
            ins = list(G.in_edges(n, data=True, keys=True))

            if len(ins) == 2 and len(outs) == 2:
                ins.sort(key=lambda x: x[0])
                outs.sort(key=lambda x: x[1])
                if ins[0][0] == outs[0][1] and ins[1][0] == outs[1][1] and ins[0][0] != outs[1][1] and (n!=ins[0][0] and n!=ins[1][0]):

                    a = ins[0][0]
                    c = ins[1][0]

                    b = n

                    for e in ins:
                        set_geometry(e)

                    for e in outs:
                        set_geometry(e)

                    l_ab = ins[0][3]
                    l_cb = ins[1][3]
                    l_ba = outs[0][3]
                    l_bc = outs[1][3]

                    l_ac = copy.deepcopy(l_ab)
                    l_ca = copy.deepcopy(l_cb)
                    gac = MultiLineString([l_ab['geometry'], l_bc['geometry']])
                    gac = ops.linemerge(gac)
                    gca = MultiLineString([l_cb['geometry'], l_ba['geometry']])
                    gca = ops.linemerge(gca)

                    l_ac['length'] = l_ab['length'] + l_bc['length']
                    l_ac['geometry'] = gac

                    l_ca['length'] = l_cb['length'] + l_ba['length']
                    l_ca['geometry'] = gca

                    G.remove_node(n)
                    G.add_edge(a, c, **l_ac)
                    G.add_edge(c, a, **l_ca)

                    to_remove.append(b)

        if len(to_remove) == 0:
            break

    print("== Removed")
    print_graph_info(G)
    return G


def simplify_graph_merge_short(G_original: nx.MultiDiGraph, threshold=100):
    '''
    Simplifies graph by merging short roads.
    :param G_original: networkx graph object.
    :param threshold: Merging threshold.
    :return: SImplified networkx graph object.
    '''
    G = G_original.copy()
    for n, info in list(G.nodes(data=True)):
        ins = list(G.in_edges(n, data=True, keys=True))
        outs = list(G.out_edges(n, data=True, keys=True))
        removed = False
        for e in ins:
            u, _, i, info = e
            if info['length'] < threshold:
                merge_nodes(G, n, u)
                G.remove_node(n)
                removed = True
                break
        if not removed:
            for e in outs:
                _, v, i, info = e
                if info['length'] < threshold:
                    merge_nodes(G, n, v)
                    G.remove_node(n)
                    break

    print("Short removed")
    print_graph_info(G)
    #print("%d to %d by remove ." % (G_original.number_of_nodes(), G.number_of_nodes()))
    return G


def merge_nodes(G, node, new_node):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """
    ins = list(G.in_edges(node, data=True, keys=True))
    outs = list(G.out_edges(node, data=True, keys=True))
    for n1, n2, _, data in ins + outs:
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 == node and n2!=new_node:
            G.add_edge(new_node, n2, **data)
        elif n2 == node and n1!=new_node:
            G.add_edge(n1, new_node, **data)


def linefy_all_geom(G_original: nx.MultiDiGraph):
    '''
    Simplifies the shape shown in the image.
    :param G_original: networkx graph object.
    :return: SImplified networkx graph object.
    '''
    G = G_original.copy()
    for e in G.edges(data=True):
        u, v, info = e
        ax = G.nodes[u]['x']
        ay = G.nodes[u]['y']
        bx = G.nodes[v]['x']
        by = G.nodes[v]['y']
        info['geometry'] = LineString([[ax, ay], [bx, by]])
    return G


# import queue
# def add_direct_edges(G_original: nx.MultiDiGraph, threshold = 2000):
#     G = G_original.copy()
#     for n in G.nodes:
#         q = queue.Queue()
#         neighbors = list(G.successors(n))
#         q.put(n)
#         #neighbors = set()
#         distances = {}
#         distances[n] = 0
#         while q.qsize() > 0:
#             v = q.get()
#             for e in list(G.out_edges(v, data=True, keys=True)):
#                 _, successor, _, info = e
#                 if (successor not in distances or distances[v] + info['length'] < distances[successor])\
#                         and distances[v] + info['length'] < threshold:
#                     q.put(successor)
#                     #neighbors.add(successor)
#                     distances[successor] = distances[v] + info['length']
#
#         for successor in distances.keys():
#             if successor not in neighbors:
#                 G.add_edge(n, successor, length=distances[successor])
#
#     print("%d to %d by add edge ." % (G_original.number_of_edges(), G.number_of_edges()))
#
#     return G
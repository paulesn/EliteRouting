import argparse
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Triangulation_3 import Delaunay_triangulation_3 as del3

import networkx as nx
import numpy as np
import math
import ijson
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
print(os.listdir("data"))

systems_preprocessed_file = "data/systems_preprocessed.edgelist"

################################################################
# Parser
################################################################
parser = argparse.ArgumentParser(
    prog='Elite Routing Preprocessing',
    description='loads a elite dangerous systems file and preproecessesthe routing graph',
    epilog='Enjoy the program! :)')

parser.add_argument('-s', '--size', type=int, default=100, help='max number of systems to load')
parser.add_argument('-i', '--input', type=str, default="data/systems.json", help='location of input file')
parser.add_argument('-o', '--output', type=str, default="data/systems_preprocessed.nx", help='location of output file')
parser.add_argument('-m', '--max_range',type=int, default=90, help='maximum jump range of the ship to reduce compute time')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-x', '--xmin', type=float, default=-1*float("inf"), help='Filter for X coord of systems')
parser.add_argument('-y', '--ymin', type=float, default=-1*float("inf"), help='Filter for Y coord of systems')
parser.add_argument('-z', '--zmin', type=float, default=-1*float("inf"), help='Filter for Z coord of systems')
parser.add_argument('-a', '--xmax', type=float, default=float("inf"), help='Filter for X coord of systems')
parser.add_argument('-b', '--ymax', type=float, default=float("inf"), help='Filter for Y coord of systems')
parser.add_argument('-c', '--zmax', type=float, default=float("inf"), help='Filter for Z coord of systems')

args = parser.parse_args()
verbose = args.verbose
input_file = args.input
output_file = args.output
max_size = args.size
max_range = args.max_range

print(f"Loading {max_size} systems from {input_file} and saving to {output_file}")


################################################################
# Utility Functions
################################################################
def d(node_u, node_v, graph):
    """
    Distance in 3D Space
    :param node_u: a node ID in the graph
    :param node_v: a node ID in the graph
    :param graph: the graph the nodes are in
    :return: the distance between the nodes
    """
    u_x, u_y, u_z = graph.nodes(data=True)[node_u]["position"]
    v_x, v_y, v_z = graph.nodes(data=True)[node_v]["position"]
    distance = math.sqrt((u_x - v_x) ** 2 + (u_y - v_y) ** 2 + (u_z - v_z) ** 2)
    return distance


def point_to_tuple(point):
    """
    Convert a CGAL point to a coordinate tuple
    :param point:
    :return:
    """
    return point.x(), point.y(), point.z()


def get_nid(x, y, z):
    """
    Get the node ID for a given coordinate
    :param x:
    :param y:
    :param z:
    :return:
    """
    nid = fac2id[x][y][z]
    if not nid is None:
        return nid
    return [nid for nid, a, b, c in points if (a == x) and (b == y) and (c == z)][0]


################################################################
# Load Data
################################################################

G = nx.Graph()
count = 0
points = []

# Open the large JSON file
with open(input_file, 'rb') as f:
    # Use ijson to parse file incrementally
    objects = ijson.items(f, 'item')
    for obj in tqdm(objects, total=max_size, desc="Loading Systems"):
        count += 1
        # Assuming each object has an 'id' and 'position' with 'x', 'y', 'z' coordinates
        node_id = obj['id64']
        position = (float(obj['coords']['x']), float(obj['coords']['y']), float(obj['coords']['z']))

        # Add node to the graph with position as attribute
        G.add_node(node_id, position=position)
        points.append([node_id, position[0], position[1], position[2]])
        if count > max_size:
            break
# store on disk
with open("data/points.npy", 'wb') as f:
    np.save(f, points, allow_pickle=False)

################################################################
# Triangulation
################################################################

G = nx.Graph()
coords = []
for nid, x, y, z in points:
    G.add_node(nid, position=(x, y, z))
    coords.append(Point_3(x, y, z))

# generate a fast access for coords 2 id
fac2id = {}

for nid, x, y, z in tqdm(points, desc="Building 2D Position Index"):
    if x not in fac2id:
        fac2id[x] = {}
    if y not in fac2id[x]:
        fac2id[x][y] = {}
    fac2id[x][y][z] = nid

dt = del3()
for coord in tqdm(coords):
    dt.insert(coord)

del coords

print("Number of cells: ", dt.number_of_cells())

for cell in tqdm(dt.finite_cells()):
    # Access the vertices of the cell (tetrahedron)
    vertices = [cell.vertex(i) for i in range(4)]

    # Add edges to the graph (each cell is a tetrahedron, so add edges accordingly)
    for i in range(4):
        for j in range(i + 1, 4):
            u = get_nid(*point_to_tuple(vertices[i].point()))
            v = get_nid(*point_to_tuple(vertices[j].point()))
            if d(u, v, G) < max_range:
                G.add_edge(u, v, distance=d(u, v, G), longest_edge=d(u, v, G))
nx.write_edgelist(G, output_file)

if G.number_of_nodes() < 100:
    posi = {nid: (pos["position"][0], pos["position"][1]) for nid, pos in G.nodes(data=True)}
    nx.draw(G, pos=posi)
    plt.show()

del points
del fac2id
del dt
nx.write_edgelist(G, systems_preprocessed_file)
################################################################
# CH calculation
################################################################
def contract(node, graph):
    """
    Contract a node in the graph
    :param node: the node to contract
    :return: edges removed, shortcuts added
    """
    removed = set()
    shortcuts = []
    for a in graph.neighbors(node):
        for b in graph.neighbors(node):
            u = min(a, b)
            v = max(a, b)
            removed.add((u, node))
            if u == v:
                continue
            sp = nx.shortest_path(graph, u, v, weight='distance')
            if node in sp:
                shortcuts.append(
                    (u, v, {
                        "longest_edge": max(
                            graph[min(u, node)][max(u, node)]["longest_edge"],
                            graph[min(v, node)][max(v, node)]["longest_edge"])
                    }))
    return removed, shortcuts


def edge_difference(node, graph):
    """
    Calculate the effect this contraction would have
    :param node: a node in the graph
    :return: edges removed - shortcuts added
    """

    removed, shortcuts = contract(node, graph)
    ed =  len(removed) - len(shortcuts)
    return ed

current_level = 1
level_map = {}
all_shortcuts = []
update_edge_diff = set(G.nodes())
print("Starting Contraction")
while G.number_of_nodes() > 1:
    ed_map = {}
    minimum = float('inf')
    for node in tqdm(update_edge_diff, desc="Calculating Edge Differences"):
        ed_map[node] = edge_difference(node, G)
        if ed_map[node] < minimum:
            minimum = ed_map[node]
            node = node
    this_iteration = [k for k, v in ed_map.items() if v == minimum]
    independent_set = nx.maximal_independent_set(G.subgraph(this_iteration))
    update_edge_diff = set()
    for node in tqdm(independent_set, desc="Contracting Nodes"):
        remove, shortcuts = contract(node,G)
        all_shortcuts.extend(shortcuts)
        for n in G.neighbors(node):
            update_edge_diff.add(n)
        G.add_edges_from(shortcuts)
        G.remove_node(node)
        level_map[node] = current_level
    current_level += 1
    print(f"\nRemoved {len(independent_set)} nodes, added {len(shortcuts)} shortcuts on level {current_level}")
    print(f"{G.number_of_nodes()} nodes remaining")
    print("--------------------------------------------------------------------------")

level_map[list(G.nodes())[0]] = current_level
del G
final_graph = nx.read_edgelist(systems_preprocessed_file)
for u, v, w in tqdm(all_shortcuts):
    final_graph.add_edge(u, v, distance=w)
for node in tqdm(final_graph.nodes()):
    final_graph.nodes[node]["level"] = level_map[node]
nx.write_edgelist(final_graph, "data/systems_preprocessed_final.edgelist")



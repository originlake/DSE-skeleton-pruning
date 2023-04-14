import sknw
import numpy as np
from skimage.draw import line
from .dse_helper import recnstrc_by_disk, get_weight

def flatten(l):
    return [item for sublist in l for item in sublist]

def _remove_branch_by_DSE(G, recn, dist, max_px_weight, checked_terminal=set()):
    deg = dict(G.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    edges = list(G.edges())
    # temporary branch reconstruction mask
    branch_recn = np.zeros_like(recn, dtype=np.int32)
    branch_recn = np.zeros_like(recn, dtype=np.int32)
    for s, e in edges:
        if s == e:
            G.remove_edge(s, e)
            continue
        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s not in terminal_points and e not in terminal_points:
                continue
            if s in checked_terminal or e in checked_terminal:
                continue
            pts = val.get('pts').tolist()
            pts.append(G.nodes[s]['o'].astype(np.int32).tolist())
            pts.append(G.nodes[e]['o'].astype(np.int32).tolist())
            recnstrc_by_disk(np.array(pts, dtype=np.int32), dist, branch_recn)
            weight = get_weight(recn, branch_recn)
            if s in terminal_points:
                checked_terminal.add(s)
                if weight < max_px_weight:
                    G.remove_node(s)
                    recn = recn - branch_recn
            if e in terminal_points:
                checked_terminal.add(e)
                if weight < max_px_weight:
                    G.remove_node(e)
                    recn = recn - branch_recn
    return G, recn

def _remove_mid_node(G):
    start_index = 0
    while True:
        nodes = [x for x in G.nodes() if G.degree(x) == 2]
        if len(nodes) == start_index:
            break
        i = nodes[start_index]
        nbs = list(G[i])
        # assert len(nbs)==2, 'degree not match'
        if len(nbs) != 2:
            start_index = start_index + 1
            continue

        edge1 = G[i][nbs[0]][0]
        edge2 = G[i][nbs[1]][0]

        s1, e1 = edge1['pts'][0], edge1['pts'][-1]
        s2, e2 = edge2['pts'][0], edge2['pts'][-1]
        dist = np.array(list(map(np.linalg.norm, [s1-s2, e1-e2, s1-e2, s2-e1])))
        if dist.argmin() == 0:
            line = np.concatenate([edge1['pts'][::-1], [G.nodes[i]['o'].astype(np.int32)], edge2['pts']], axis=0)
        elif dist.argmin() == 1:
            line = np.concatenate([edge1['pts'], [G.nodes[i]['o'].astype(np.int32)], edge2['pts'][::-1]], axis=0)
        elif dist.argmin() == 2:
            line = np.concatenate([edge2['pts'], [G.nodes[i]['o'].astype(np.int32)], edge1['pts']], axis=0)
        elif dist.argmin() == 3:
            line = np.concatenate([edge1['pts'], [G.nodes[i]['o'].astype(np.int32)], edge2['pts']], axis=0)
        G.add_edge(nbs[0], nbs[1], weight=edge1['weight']+edge2['weight'], pts=line)
        G.remove_node(i)
    return G

def skel_pruning_DSE(skel, dist, min_area_px=100, return_graph=False):
    """Skeleton pruning using dse
    
    Arguments:
        skel {ndarray} -- skeleton obtained from skeletonization algorithm
        dist {ndarray} -- distance transfrom map
    
    Keyword Arguments:
        min_area_px {int} -- branch reconstruction weights, measured by pixel area. Branch reconstruction weights smaller than this threshold will be pruned. (default: {100})
        return_graph {bool} -- return graph

    Returns:
        ndarray -- pruned skeleton map
    """
    graph = sknw.build_sknw(skel, multi=True)
    dist = dist.astype(np.int32)
    graph = _remove_mid_node(graph)
    edges = list(set(graph.edges()))
    pts = []
    for s, e in edges:
        vals = flatten([[v] for v in graph[s][e].values()])
        for ix, val in enumerate(vals):
            pts.extend(val.get('pts').tolist())
        pts.append(graph.nodes[s]['o'].astype(np.int32).tolist())
        pts.append(graph.nodes[e]['o'].astype(np.int32).tolist()) 
    recnstrc = np.zeros_like(dist, dtype=np.int32)
    recnstrc_by_disk(np.array(pts, dtype=np.int32), dist, recnstrc) 
    num_nodes = len(graph.nodes())
    checked_terminal = set()
    while True:
        # cannot combine with other pruning method because the reconstruction map is not updated in other approach
        graph, recnstrc = _remove_branch_by_DSE(graph, recnstrc, dist, min_area_px, checked_terminal=checked_terminal)
        if len(graph.nodes()) == num_nodes:
            break
        graph = _remove_mid_node(graph)
        num_nodes = len(graph.nodes())
    if return_graph:
        return graph2im(graph, skel.shape), graph
    else:
        return graph2im(graph, skel.shape)      

def graph2im(graph, shape):
    mask = np.zeros(shape, dtype=np.bool)
    for s,e in graph.edges():
        vals = flatten([[v] for v in graph[s][e].values()])
        for val in vals:
            coords = val.get('pts')
            coords_1 = np.roll(coords, -1, axis=0)
            for i in range(len(coords)-1):
                rr, cc = line(*coords[i], *coords_1[i])
                mask[rr, cc] = True
            mask[tuple(graph.nodes[s]['pts'].T.tolist())] = True
            mask[tuple(graph.nodes[e]['pts'].T.tolist())] = True
    return mask
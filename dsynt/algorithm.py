import torch
#import mip
import cplex
from dsynt.graph import Graph

def argmax_simple(weight, node_weights=None):
    # get the label of maximum weight for each arc
    max_values, max_indices = weight.max(dim=2)
    # keep only max label arc with positive weights
    pred_arcs = (max_values > 0)
    # Erase all arcs going to the head
    pred_arcs[:, 0] = False
    # erase diagonal
    pred_arcs = pred_arcs * ~torch.eye(pred_arcs.shape[0], dtype=torch.bool, device=pred_arcs.device)

    nodes = None
    if node_weights is not None:
        nodes = 1 * (node_weights > 0).reshape(-1)
    return pred_arcs, max_indices, nodes


def argmax_outgoing_arcs(arc_weights, node_weights):
    # labels with max weight
    max_values, max_indices = arc_weights.max(dim=2)

    # select positive arcs
    pred_arcs = (max_values > 0)
    # Erase all arcs going to the head
    pred_arcs[:, 0] = False
    # erase diagonal
    pred_arcs = pred_arcs * ~torch.eye(pred_arcs.shape[0], dtype=torch.bool, device=pred_arcs.device)

    # sum of positive arcs for each node
    outgoing_score = (max_values * pred_arcs).sum(dim=1)
    # add node weights
    node_weights = node_weights.reshape(-1)
    node_weights2 = torch.zeros((node_weights.shape[0] + 1,), requires_grad=False, device=node_weights.device)
    node_weights2[1:] = node_weights
    node_score = outgoing_score + node_weights2
    # selected nodes are nodes with positive score
    selected_nodes = (node_score > 0)
    selected_nodes[0] = True

    # select correct pred arcs
    pred_arcs = pred_arcs * selected_nodes.unsqueeze(1)

    return pred_arcs, max_indices, selected_nodes[1:]


def is_connected(arcs):
    arcs = arcs.cpu()

    edges = {}
    for i2 in range(arcs.shape[0]):
        for j2 in range(1, arcs.shape[0]):
            if i2 == j2:
                continue
            if not arcs[i2, j2]:
                continue

            i, j = min(i2, j2), max(i2, j2)
            if i in edges:
                edges[i].append(j)
            else:
                edges[i] = [j]
            if j in edges:
                edges[j].append(i)
            else:
                edges[j] = [i]

    visited = set()
    stack = set()
    stack.add(next(iter(edges)))
    while len(stack) > 0:
        current = stack.pop()
        visited.add(current)
        for other in edges[current]:
            if other not in visited:
                stack.add(other)

    return len(visited) == len(edges) and 0 in edges


def argmax_semi_structured(weights, node_weights, linear_relaxation=False):
    n_words = node_weights.shape[0]

    # get the label of maximum weight for each arc
    max_values, max_indices = weights.max(dim=2)

    # transform arc weights into edge weights
    # the edge weights between a and b is degined as follows:
    # w(a, b) > 0 and w(b, a) > 0: w(a, b) + w(b, a)
    # w(a, b) <= 0 and w(b, a) > 0: w(b, a)
    # w(a, b) > 0 and w(b, a) <= 0: w(a, b)
    # w(a, b) <= 0 and w(b, a) <= 0: max(w(a, b), w(b, a))
    t_arc_weights = max_values.transpose(0, 1)
    # erase weight going to the root (first ligne because transpose)
    t_arc_weights[0, :] = -float("inf")
    edge_weights = torch.max(max_values + t_arc_weights, torch.max(max_values, t_arc_weights))

    max_values = max_values.cpu()  # required to compute output
    edge_weights = edge_weights.cpu()
    node_weights = node_weights.cpu()

    graph = Graph(
        n_words + 1,
        lambda i: 0 if i == 0 else node_weights[i - 1].item(),
        lambda i, j: edge_weights[i, j].item()
    )
    # Reduction 1:
    # if w(a, b) > and w(a) > and w(b) > 0,
    # the we can contract into a single node which is always selected
    while True:
        has_merged_something = False
        for i, j in graph.arc_iterator():
            if graph.arc_weight(i, j) > 0 and graph.arc_weight(i, j) + graph.node_weight(i) > 0 and graph.arc_weight(i, j) + graph.node_weight(j) > 0:
                graph.merge(i, j)
                has_merged_something = True
                break
        if not has_merged_something:
            break

    """
    # Reduction 2:
    # if w(a, b) <= 0 and w(a) <= 0 and w(b) <= 0,
    # we can remove w(a, b)
    to_remove = list()
    for i, j in graph.arc_iterator():
        if graph.arc_weight(i, j) <= 0 and graph.node_weight(i) <= 0 and graph.node_weight(j) <= 0:
            to_remove.append((i, j))
    for i, j in to_remove:
        graph.remove_arc(i, j)
    """


    if graph.n_nodes() == 1:
        pred_arcs = torch.zeros_like(max_values, dtype=torch.bool, device="cpu", requires_grad=False)
        selected_nodes = torch.zeros(n_words, dtype=torch.bool, device="cpu", requires_grad=False)

        o_nodes, o_arcs = graph.get_original_from_node(next(graph.node_iterator()))
        for o in o_nodes:
            if o > 0:
                selected_nodes[o - 1] = True
        for i2, j2 in o_arcs:
            if i2 == 0:
                pred_arcs[i2, j2] = True
            elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                pred_arcs[i2, j2] = True
                pred_arcs[j2, i2] = True
            elif max_values[i2, j2] > max_values[j2, i2]:
                pred_arcs[i2, j2] = True
            else:
                pred_arcs[j2, i2] = True

        return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)
    else:
        mprog = cplex.Cplex()
        mprog.set_log_stream(None)
        mprog.set_error_stream(None)
        mprog.set_warning_stream(None)
        mprog.set_results_stream(None)

        mprog.objective.set_sense(mprog.objective.sense.maximize)

        variables = ["x_" + str(i) for i in graph.node_iterator()]
        scores = [graph.node_weight(i) for i in graph.node_iterator()]
        # map node to var index
        nodes = {n: i for i, n in enumerate(graph.node_iterator())}

        arcs = dict()
        for i, j in graph.arc_iterator():
            name = "y_" + str(i) + "_" + str(j)
            arcs[(i, j)] = len(variables)
            variables.append(name)
            scores.append(graph.arc_weight(i, j))

        lb = [0] * len(variables)
        ub = [1] * len(variables)

        x = mprog.variables.add(
            obj=scores,
            lb=lb,
            ub=ub,
            names=variables,
            types="" if linear_relaxation else [mprog.variables.type.binary] * len(scores)
        )

        constraing_lhd = []
        constraing_rhs = []
        constraing_sign = []

        # forced merged nodes to 1
        for i in graph.node_iterator():
            if i >= graph.original_size:
                constraing_lhd.append([
                    [nodes[i]],
                    [1.]
                ])
                constraing_rhs.append(1)
                constraing_sign.append("E")

        # each node is forced to 1 if at least one adjacent arc
        for i in graph.node_iterator():
            for i2, j2 in graph.adj_arcs[i].keys():
                # node >= arcs[(head, i + 1)]
                constraing_lhd.append([
                    [nodes[i], arcs[(i2, j2)]],
                    [1., -1.]
                ])
                constraing_rhs.append(0)
                constraing_sign.append("G")

        # each node is forced to 0 if no adjacent arc
        for i in graph.node_iterator():
            # node <= mip.xsum(adjacent_arcs)
            adjacent_arcs = [arcs[(i2, j2)] for i2, j2 in graph.adj_arcs[i].keys()]
            if len(adjacent_arcs) == 0:
                continue
            constraing_lhd.append([
                [nodes[i]] + adjacent_arcs,
                [1.] + [-1.] * len(adjacent_arcs)
            ])
            constraing_rhs.append(0)
            constraing_sign.append("L")

        mprog.linear_constraints.add(
            lin_expr=constraing_lhd,
            senses=constraing_sign,
            rhs=constraing_rhs,
        )

        # Solve the problem
        mprog.solve()

        # And print the solutions
        variable_values = mprog.solution.get_values()

        if linear_relaxation:
            pred_arcs = torch.zeros_like(max_values, dtype=torch.float, device="cpu", requires_grad=False)
            selected_nodes = torch.zeros(n_words, dtype=torch.float, device="cpu", requires_grad=False)

            for (head, mod), v in arcs.items():
                v = variable_values[v]
                for i2, j2 in graph.get_original_from_arc(head, mod):
                    if i2 == 0:
                        pred_arcs[i2, j2] = v
                    elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                        pred_arcs[i2, j2] = v
                        pred_arcs[j2, i2] = v
                    elif max_values[i2, j2] > max_values[j2, i2]:
                        pred_arcs[i2, j2] = v
                    else:
                        pred_arcs[j2, i2] = v

            for n, i in nodes.items():
                v = variable_values[i]
                o_nodes, o_arcs = graph.get_original_from_node(n)
                for o in o_nodes:
                    if o > 0:
                        selected_nodes[o - 1] = v
                for i2, j2 in o_arcs:
                    if i2 == 0:
                        pred_arcs[i2, j2] = v
                    elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                        pred_arcs[i2, j2] = v
                        pred_arcs[j2, i2] = v
                    elif max_values[i2, j2] > max_values[j2, i2]:
                        pred_arcs[i2, j2] = v
                    else:
                        pred_arcs[j2, i2] = v

            #print(pred_arcs)
            #print(selected_nodes)
            return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)
        else:
            pred_arcs = torch.zeros_like(max_values, dtype=torch.bool, device="cpu", requires_grad=False)
            selected_nodes = torch.zeros(n_words, dtype=torch.bool, device="cpu", requires_grad=False)

            for (head, mod), v in arcs.items():
                if variable_values[v] > 0.99:
                    for i2, j2 in graph.get_original_from_arc(head, mod):
                        if i2 == 0:
                            pred_arcs[i2, j2] = True
                        elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                            pred_arcs[i2, j2] = True
                            pred_arcs[j2, i2] = True
                        elif max_values[i2, j2] > max_values[j2, i2]:
                            pred_arcs[i2, j2] = True
                        else:
                            pred_arcs[j2, i2] = True

            for n, i in nodes.items():
                if variable_values[i] > 0.99:
                    o_nodes, o_arcs = graph.get_original_from_node(n)
                    for o in o_nodes:
                        if o > 0:
                            selected_nodes[o - 1] = True
                    for i2, j2 in o_arcs:
                        if i2 == 0:
                            pred_arcs[i2, j2] = True
                        elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                            pred_arcs[i2, j2] = True
                            pred_arcs[j2, i2] = True
                        elif max_values[i2, j2] > max_values[j2, i2]:
                            pred_arcs[i2, j2] = True
                        else:
                            pred_arcs[j2, i2] = True

            #print(pred_arcs)
            #print(selected_nodes)
            return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)




def argmax_structured(weights, node_weights, linear_relaxation=False):
    #if node_weights is not None:
    #    raise RuntimeError("Node weights not supported anymore")

    n_words = node_weights.shape[0]

    # get the label of maximum weight for each arc
    max_values, max_indices = weights.max(dim=2)

    # we first try to select all positive arcs,
    # if the structure is connected, then it is the optimal solution
    #simple_pred_arcs = (max_values > 0)
    #simple_pred_arcs[:, 0] = False
    #pred_arcs = simple_pred_arcs * ~torch.eye(simple_pred_arcs.shape[0], dtype=torch.bool, device=simple_pred_arcs.device)
    #if is_connected(pred_arcs):
    #    return pred_arcs, max_indices, None

    # transform arc weights into edge weights
    # the edge weights between a and b is degined as follows:
    # w(a, b) > 0 and w(b, a) > 0: w(a, b) + w(b, a)
    # w(a, b) <= 0 and w(b, a) > 0: w(b, a)
    # w(a, b) > 0 and w(b, a) <= 0: w(a, b)
    # w(a, b) <= 0 and w(b, a) <= 0: max(w(a, b), w(b, a))
    t_arc_weights = max_values.transpose(0, 1)
    # erase weight going to the root (first ligne because transpose)
    t_arc_weights[0, :] = -float("inf")
    edge_weights = torch.max(max_values + t_arc_weights, torch.max(max_values, t_arc_weights))

    max_values = max_values.cpu()  # required to compute output
    edge_weights = edge_weights.cpu()
    node_weights = node_weights.cpu()

    graph = Graph(
        n_words + 1,
        lambda i: 0 if i == 0 else node_weights[i - 1].item(),
        lambda i, j: edge_weights[i, j].item()
    )
    # Reduction 1:
    # if w(a, b) > and w(a) > and w(b) > 0,
    # the we can contract into a single node which is always selected
    while True:
        has_merged_something = False
        for i, j in graph.arc_iterator():
            if graph.arc_weight(i, j) > 0 and graph.arc_weight(i, j) + graph.node_weight(i) > 0 and graph.arc_weight(i, j) + graph.node_weight(j) > 0:
                graph.merge(i, j)
                has_merged_something = True
                break
        if not has_merged_something:
            break

    if graph.n_nodes() == 1:
        pred_arcs = torch.zeros_like(max_values, dtype=torch.bool, device="cpu", requires_grad=False)
        selected_nodes = torch.zeros(n_words, dtype=torch.bool, device="cpu", requires_grad=False)

        o_nodes, o_arcs = graph.get_original_from_node(next(graph.node_iterator()))
        for o in o_nodes:
            if o > 0:
                selected_nodes[o - 1] = True
        for i2, j2 in o_arcs:
            if i2 == 0:
                pred_arcs[i2, j2] = True
            elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                pred_arcs[i2, j2] = True
                pred_arcs[j2, i2] = True
            elif max_values[i2, j2] > max_values[j2, i2]:
                pred_arcs[i2, j2] = True
            else:
                pred_arcs[j2, i2] = True

        return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)
    else:
        if graph.root == -1:
            raise RuntimeError("Root disappeared... :(")

        mprog = cplex.Cplex()
        mprog.set_log_stream(None)
        mprog.set_error_stream(None)
        mprog.set_warning_stream(None)
        mprog.set_results_stream(None)

        mprog.objective.set_sense(mprog.objective.sense.maximize)

        variables = ["x_" + str(i) for i in graph.node_iterator()]
        vtypes = [mprog.variables.type.binary] * len(variables)
        lb = [0] * len(variables)
        ub = [1] * len(variables)
        scores = [graph.node_weight(i) for i in graph.node_iterator()]
        # map node to var index
        nodes = {n: i for i, n in enumerate(graph.node_iterator())}

        # distance variables
        distances = dict()
        for n in nodes.keys():
            if n != graph.root:
                distances[n] = len(variables)
                variables.append("d_" + str(n))
                scores.append(0.)
                vtypes.append(mprog.variables.type.continuous)
                lb.append(2.)
                ub.append(graph.n_nodes())

        # edges of the "real graph"
        edges = dict()
        for i, j in graph.arc_iterator():
            name = "y_" + str(i) + "_" + str(j)
            edges[(i, j)] = len(variables)
            variables.append(name)
            scores.append(graph.arc_weight(i, j))
            vtypes.append(mprog.variables.type.binary)
            lb.append(0.)
            ub.append(1.)

        # arcs that ensure connectedness
        arcs = dict()
        for i, j in graph.arc_iterator():
            if j != graph.root:
                name = "a_" + str(i) + "_" + str(j)
                arcs[(i, j)] = len(variables)
                variables.append(name)
                scores.append(0.)
                vtypes.append(mprog.variables.type.binary)
                lb.append(0.)
                ub.append(1.)
            if i != graph.root:
                name = "a_" + str(j) + "_" + str(i)
                arcs[(j, i)] = len(variables)
                variables.append(name)
                scores.append(0.)
                vtypes.append(mprog.variables.type.binary)
                lb.append(0.)
                ub.append(1.)

        x = mprog.variables.add(
            obj=scores,
            lb=lb,
            ub=ub,
            names=variables,
            types="" if linear_relaxation else vtypes
        )

        constraing_lhd = []
        constraing_rhs = []
        constraing_sign = []


        # constraint (1)
        # w_e <= y_b
        for node, var_n in nodes.items():
            for i, j in graph.adj_arcs[node].keys():
                constraing_lhd.append([
                    [edges[(i, j)], var_n],
                    [1., -1.]
                ])
                constraing_rhs.append(0)
                constraing_sign.append("L")

        # arc can be selected only if the corresponding edge is selected
        # constraint (5)
        for (i, j), e in edges.items():
            if i == graph.root:
                constraing_lhd.append([
                    [arcs[(i, j)], e],
                    [1., -1.]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(0)
            elif j == graph.root:
                constraing_lhd.append([
                    [arcs[(j, i)], e],
                    [1., -1.]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(0)
            else:
                constraing_lhd.append([
                    [arcs[(i, j)], arcs[(j, i)], e],
                    [1., 1., -1.]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(0)

        # constraint on distance variable domain,
        # we start at 2 because the root is fixed
        # constraint (3)
        # Useless => ensured by bounds
        """
        for d in distances.values():
            constraing_lhd.append([
                [d],
                [1.]
            ])
            constraing_sign.append("G")
            constraing_rhs.append(2.)

            constraing_lhd.append([
                [d],
                [1.]
            ])
            constraing_sign.append("L")
            constraing_rhs.append(graph.n_nodes())
        """

        # if a node is selected, it must have exactly one incoming arc
        # (except for the root node)
        # constraint (4)
        for n, var_n in nodes.items():
            if n == graph.root:
                continue
            v = [var_a for (i, j), var_a in arcs.items() if j == n]
            constraing_lhd.append([
                v + [var_n],
                [1.] * len(v) + [-1.]
            ])
            constraing_sign.append("E")
            constraing_rhs.append(0.)

        # (non-linear) distance constraints!
        # constraints (9) and (10)
        for (i, j), var_a in arcs.items():
            # (9) n + d_j - d_i >= (n + 1) a_ij
            # (10) n + d_i - d_j >= (n - 1) a_ij
            if i == graph.root:
                # root has no distance, it is implicityly set to one:
                # (9) n + d_j - 1 >= (n + 1) a_ij
                # d_j - (n + 1) a_ij >= 1 - n
                constraing_lhd.append([
                    [distances[j], arcs[(i, j)]],
                    [1., -(graph.n_nodes() + 1)]
                ])
                constraing_sign.append("G")
                constraing_rhs.append(1 - graph.n_nodes())

                # (10) n + 1 - d_j >= (n - 1) a_ij
                # -d_j - (n - 1) a_ij >= -1 - n
                constraing_lhd.append([
                    [distances[j], arcs[(i, j)]],
                    [-1., -(graph.n_nodes() - 1)]
                ])
                constraing_sign.append("G")
                constraing_rhs.append(-1 - graph.n_nodes())
            else:
                # (9) n + d_j - d_i >= (n + 1) a_ij
                # d_j - d_i - (n + 1) a_ij >= - n
                constraing_lhd.append([
                    [distances[j], distances[i], arcs[(i, j)]],
                    [1., -1., -(graph.n_nodes() + 1)]
                ])
                constraing_sign.append("G")
                constraing_rhs.append(-graph.n_nodes())

                # (10) n + d_i - d_j >= (n - 1) a_ij
                # d_i - d_j - (n - 1) a_ij >= -n
                constraing_lhd.append([
                    [distances[i], distances[j], arcs[(i, j)]],
                    [1., -1., -(graph.n_nodes() - 1)]
                ])
                constraing_sign.append("G")
                constraing_rhs.append(-graph.n_nodes())


        # supplementary constraints (12) (13)
        for (v, u), var_e in edges.items():
            # (12) d_v - d_u <= n - (n - 1) e_vu
            # (13) d_u - d_v <= n - (n - 1) e_vu
            if v == graph.root:
                # (12) 1 - d_u <= n - (n - 1) e_vu
                # - d_u + (n - 1) e_vu <= n - 1
                constraing_lhd.append([
                    [distances[u], var_e],
                    [-1., (graph.n_nodes() - 1)]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(graph.n_nodes() - 1)

                # (13) d_u - 1 <= n - (n - 1) e_vu
                # d_u + (n - 1) e_vu <= n + 1
                constraing_lhd.append([
                    [distances[u], var_e],
                    [1., (graph.n_nodes() - 1)]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(graph.n_nodes() + 1)
            elif u == graph.root:
                # (12) d_v - 1 <= n - (n - 1) e_vu
                # d_v + (n - 1) e_vu <= n + 1
                constraing_lhd.append([
                    [distances[v], var_e],
                    [1., (graph.n_nodes() - 1)]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(graph.n_nodes() + 1)

                # (13) 1 - d_v <= n - (n - 1) e_vu
                # - d_v + (n - 1) e_vu <= n - 1
                constraing_lhd.append([
                    [distances[v], var_e],
                    [-1., (graph.n_nodes() - 1)]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(graph.n_nodes() - 1)
            else:
                # (12) d_v - d_u <= n - (n - 1) e_vu
                # d_v - d_u + (n - 1) e_vu <= n
                constraing_lhd.append([
                    [distances[v], distances[u], var_e],
                    [1., -1., (graph.n_nodes() - 1)]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(graph.n_nodes())

                # (13) d_u - d_v <= n - (n - 1) e_vu
                # d_u - d_v + (n - 1) e_vu <= n
                constraing_lhd.append([
                    [distances[u], distances[v], var_e],
                    [1., -1., (graph.n_nodes() - 1)]
                ])
                constraing_sign.append("L")
                constraing_rhs.append(graph.n_nodes())

        mprog.linear_constraints.add(
            lin_expr=constraing_lhd,
            senses=constraing_sign,
            rhs=constraing_rhs,
        )

        # Solve the problem
        mprog.solve()

        # And print the solutions
        variable_values = mprog.solution.get_values()

        if linear_relaxation:
            pred_arcs = torch.zeros_like(max_values, dtype=torch.float, device="cpu", requires_grad=False)
            selected_nodes = torch.zeros(n_words, dtype=torch.float, device="cpu", requires_grad=False)

            for (head, mod), v in edges.items():
                v = variable_values[v]
                for i2, j2 in graph.get_original_from_arc(head, mod):
                    if i2 == 0:
                        pred_arcs[i2, j2] = v
                    elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                        pred_arcs[i2, j2] = v
                        pred_arcs[j2, i2] = v
                    elif max_values[i2, j2] > max_values[j2, i2]:
                        pred_arcs[i2, j2] = v
                    else:
                        pred_arcs[j2, i2] = v

            for n, i in nodes.items():
                v = variable_values[i]
                o_nodes, o_arcs = graph.get_original_from_node(n)
                for o in o_nodes:
                    if o > 0:
                        selected_nodes[o - 1] = v
                for i2, j2 in o_arcs:
                    if i2 == 0:
                        pred_arcs[i2, j2] = v
                    elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                        pred_arcs[i2, j2] = v
                        pred_arcs[j2, i2] = v
                    elif max_values[i2, j2] > max_values[j2, i2]:
                        pred_arcs[i2, j2] = v
                    else:
                        pred_arcs[j2, i2] = v

            #print(pred_arcs)
            #print(selected_nodes)
            return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)
        else:
            pred_arcs = torch.zeros_like(max_values, dtype=torch.bool, device="cpu", requires_grad=False)
            selected_nodes = torch.zeros(n_words, dtype=torch.bool, device="cpu", requires_grad=False)

            for (head, mod), v in edges.items():
                if variable_values[v] > 0.99:
                    for i2, j2 in graph.get_original_from_arc(head, mod):
                        if i2 == 0:
                            pred_arcs[i2, j2] = True
                        elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                            pred_arcs[i2, j2] = True
                            pred_arcs[j2, i2] = True
                        elif max_values[i2, j2] > max_values[j2, i2]:
                            pred_arcs[i2, j2] = True
                        else:
                            pred_arcs[j2, i2] = True

            for n, i in nodes.items():
                if variable_values[i] > 0.99:
                    o_nodes, o_arcs = graph.get_original_from_node(n)
                    for o in o_nodes:
                        if o > 0:
                            selected_nodes[o - 1] = True
                    for i2, j2 in o_arcs:
                        if i2 == 0:
                            pred_arcs[i2, j2] = True
                        elif max_values[i2, j2] > 0 and max_values[j2, i2] > 0:
                            pred_arcs[i2, j2] = True
                            pred_arcs[j2, i2] = True
                        elif max_values[i2, j2] > max_values[j2, i2]:
                            pred_arcs[i2, j2] = True
                        else:
                            pred_arcs[j2, i2] = True

            #print(pred_arcs)
            #print(selected_nodes)
            return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)


"""

def argmax_semi_structured(weights, node_weights, linear_relaxation=False):
    n_words = node_weights.shape[0]

    # get the label of maximum weight for each arc
    arc_weights, max_indices = weights.max(dim=2)

    # transform arc weights into edge weights
    # the edge weights between a and b is degined as follows:
    # w(a, b) > 0 and w(b, a) > 0: w(a, b) + w(b, a)
    # w(a, b) <= 0 and w(b, a) > 0: w(b, a)
    # w(a, b) > 0 and w(b, a) <= 0: w(a, b)
    # w(a, b) <= 0 and w(b, a) <= 0: max(w(a, b), w(b, a))
    t_arc_weights = arc_weights.transpose(0, 1)
    edge_weights = torch.max(arc_weights + t_arc_weights, torch.max(arc_weights, t_arc_weights))

    edge_weights = edge_weights.cpu()
    node_weights = node_weights.cpu()

    graph = Graph(
        n_words + 1,
        lambda i: 0 if i == 0 else node_weights[i - 1].item(),
        lambda i, j: weights[i, j].item()
    )
    # Reduction:
    # if w(a, b) + w(a) + w(b) > 0,
    # the we can contract into a single node which is always selected
    while True:
        has_merged_something = False
        for i, j in graph.arc_iterator():
            w = graph.arc_weight(i, j) + graph.node_weight(i) + graph.node_weight(j)
            if w > 0:
                graph.merge(i, j)
                has_merged_something = True
                break
        if not has_merged_something:
            break


    mprog = cplex.Cplex()
    mprog.set_log_stream(None)
    mprog.set_error_stream(None)
    mprog.set_warning_stream(None)
    mprog.set_results_stream(None)

    mprog.objective.set_sense(mprog.objective.sense.maximize)

    variables = ["x_" + str(i + 1) for i in range(n_words)]
    scores = [node_weights[i].item() for i in range(n_words)]

    arcs = dict()
    for head in range(n_words + 1):
        for mod in range(1, n_words + 1):
            if head != mod:
                name = "y_" + str(head) + "_" + str(mod)
                arcs[(head, mod)] = len(variables)
                variables.append(name)
                scores.append(arc_weights[head, mod].item())

    lb = [0] * len(variables)
    ub = [1] * len(variables)

    x = mprog.variables.add(
        obj=scores,
        lb=lb,
        ub=ub,
        names=variables,
        types="" if linear_relaxation else [mprog.variables.type.binary] * len(scores)
    )

    constraing_lhd = []
    constraing_rhs = []
    constraing_sign = []

    # each node is forced to 1 if at least one adjacent arc
    for i in range(n_words):
        for head in range(0, n_words + 1):
            if head == i + 1:
                continue
            # node >= arcs[(head, i + 1)]
            constraing_lhd.append([
                [i, arcs[(head, i + 1)]],
                [1., -1.]
            ])
            constraing_rhs.append(0)
            constraing_sign.append("G")

        for mod in range(1, n_words + 1):
            if mod == i + 1:
                continue
            # node >= arcs[(i + 1, mod)]
            constraing_lhd.append([
                [i, arcs[(i + 1, mod)]],
                [1., -1.]
            ])
            constraing_rhs.append(0)
            constraing_sign.append("G")

    # each node is forced to 0 if no adjacent arc
    for i in range(n_words):
        adjacent_arcs = [i]
        adjacent_arcs.extend(
            arcs[(head, i + 1)]
            for head in range(0, n_words + 1)
            if head != i + 1
        )
        adjacent_arcs.extend(
            arcs[(i + 1, mod)]
            for mod in range(1, n_words + 1)
            if mod != i + 1
        )
        # node <= mip.xsum(adjacent_arcs)
        constraing_lhd.append([
            adjacent_arcs,
            [1.] + [-1.] * (len(adjacent_arcs) - 1)
        ])
        constraing_rhs.append(0)
        constraing_sign.append("L")

    mprog.linear_constraints.add(
        lin_expr=constraing_lhd,
        senses=constraing_sign,
        rhs=constraing_rhs,
    )

    # Solve the problem
    mprog.solve()

    # And print the solutions
    variable_values = mprog.solution.get_values()

    pred_arcs = torch.zeros_like(arc_weights, dtype=bool, device="cpu", requires_grad=False)
    for (head, mod), v in arcs.items():
        if variable_values[v] > 0.99:
            pred_arcs[head, mod] = True

    selected_nodes = torch.zeros(n_words, dtype=bool, device="cpu", requires_grad=False)
    for i in range(n_words):
        if variable_values[i] > 0.99:
            selected_nodes[i] = True

    #print(pred_arcs)
    #print(selected_nodes)
    return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)



MIP + Gurobi implementation,
unfortunately I cannot run Gurobi on the cluster due to licensing problems
def argmax_semi_structured(weights, node_weights):
    # get the label of maximum weight for each arc
    arc_weights, max_indices = weights.max(dim=2)
    arc_weights = arc_weights.cpu()
    node_weights = node_weights.cpu()

    m = mip.Model('knapsack', sense= mip.MAXIMIZE, solver_name=mip.GRB)
    m.verbose = 0

    nodes = [m.add_var(var_type=mip.BINARY) for _ in range(node_weights.shape[0])]
    score_items = [node_weights[i].item() * nodes[i] for i in range(len(nodes))]

    arcs = dict()
    for head in range(node_weights.shape[0] + 1):
        for mod in range(1, node_weights.shape[0] + 1):
            if head != mod:
                v = m.add_var(var_type=mip.BINARY)
                arcs[(head, mod)] = v
                score_items.append(v * arc_weights[head, mod].item())

    m.objective = mip.maximize(mip.xsum(score_items))

    # each node is forced to 1 if at least one adjacent arc
    for i, node in enumerate(nodes):
        for head in range(0, len(nodes) + 1):
            if head == i + 1:
                continue
            m += node -  arcs[(head, i + 1)] >= 0
        for mod in range(1, len(nodes) + 1):
            if mod == i + 1:
                continue
            m += node - arcs[(i + 1, mod)] >= 0

    # each node is forced to 0 if no adjacent arc
    for i, node in enumerate(nodes):
        adjacent_arcs = [arcs[(head, i + 1)] for head in range(0, len(nodes) + 1) if head != i + 1]
        adjacent_arcs += [arcs[(i + 1, mod)] for mod in range(1, len(nodes) + 1) if mod != i + 1]
        m += node <= mip.xsum(adjacent_arcs)

    status = m.optimize()

    pred_arcs = torch.zeros_like(arc_weights, dtype=bool, device="cpu", requires_grad=False)
    for (head, mod), v in arcs.items():
        if v.x > 0.99:
            pred_arcs[head, mod] = True

    selected_nodes = torch.zeros(len(nodes), dtype=bool, device="cpu", requires_grad=False)
    for i, x in enumerate(nodes):
        if x.x > 0.99:
            selected_nodes[i] = True

    #print(pred_arcs)
    #print(selected_nodes)
    return pred_arcs.to(max_indices.device), max_indices, selected_nodes.to(max_indices.device)
"""

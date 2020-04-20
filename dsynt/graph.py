
def iter_arcs(graph):
    for i in graph.nodes.keys():
        for i2, j2 in graph.adj_arcs[i].keys():
            if i2 == i:  # we need a condition like this so we produce each arc only once
                yield i2, j2


class Graph:
    def __init__(self, size, node_weights, arc_weights):
        self.adj_arcs = {i: dict() for i in range(size)}
        self.nodes = dict()
        self.original_size = size

        self.node_to_original_nodes = dict()
        self.node_to_original_arcs = dict()
        self.arc_to_original = dict()

        self.last_node = size - 1

        for i in range(size):
            self.nodes[i] = node_weights(i)
            for j in range(i + 1, size):
                w = arc_weights(i, j)
                self.adj_arcs[i][i, j] = w
                self.adj_arcs[j][i, j] = w
        self.root = 0

    def n_nodes(self):
        return len(self.nodes)

    def arc_iterator(self):
        return iter_arcs(self)

    def node_iterator(self):
        return iter(self.nodes.keys())

    def arc_weight(self, i, j):
        return self.adj_arcs[i][i, j]

    def node_weight(self, node):
        return self.nodes[node]

    def remove_arc(self, i, j):
        # remove arc from graph
        del self.adj_arcs[i][i, j]
        del self.adj_arcs[j][i, j]

        # remove nodes from the graph if no adjacent edge anymore
        if len(self.adj_arcs[i]) == 0 and i < self.original_size:
            del self.adj_arcs[i]
            del self.nodes[i]
            if self.root == i:
                self.root = -1
        if len(self.adj_arcs[j]) == 0 and j < self.original_size:
            del self.adj_arcs[j]
            del self.nodes[j]
            if self.root == j:
                self.root = -1

    def merge(self, i, j):
        new_node_weight = self.nodes[i] + self.nodes[j] + self.adj_arcs[i][i, j]
        new_node_id = self.get_new_node_id()
        if self.root == i or self.root == j:
            self.root = new_node_id
        self.nodes[new_node_id] = new_node_weight
        self.adj_arcs[new_node_id] = dict()

        if i < self.original_size and j < self.original_size:
            # this are two "original nodes"
            self.node_to_original_arcs[new_node_id] = [(i, j)]
        elif i >= self.original_size and j >= self.original_size:
            # this are two "merged nodes"
            self.node_to_original_arcs[new_node_id] = self.node_to_original_arcs[i] + self.node_to_original_arcs[j] + self.arc_to_original[(i, j)]
            del self.node_to_original_arcs[i]
            del self.node_to_original_arcs[j]
            del self.arc_to_original[(i, j)]
        elif i < self.original_size:
            self.node_to_original_arcs[new_node_id] = self.node_to_original_arcs[j] + self.arc_to_original[(i, j)]
            del self.node_to_original_arcs[j]
            del self.arc_to_original[(i, j)]
        else:
            self.node_to_original_arcs[new_node_id] = self.node_to_original_arcs[i] + self.arc_to_original[(i, j)]
            del self.node_to_original_arcs[i]
            del self.arc_to_original[(i, j)]

        mapped_nodes = []
        if i in self.node_to_original_nodes:
            mapped_nodes.extend(self.node_to_original_nodes[i])
            del self.node_to_original_nodes[i]
        else:
            mapped_nodes.append(i)
        if j in self.node_to_original_nodes:
            mapped_nodes.extend(self.node_to_original_nodes[j])
            del self.node_to_original_nodes[j]
        else:
            mapped_nodes.append(j)
        self.node_to_original_nodes[new_node_id] = mapped_nodes

        # we need to remap all adjacent arcs from i or j
        for i2, j2 in self.adj_arcs[i]:
            other = j2 if i2 == i else i2
            if other == j:
                continue
            arc = (i, other) if i < other else (other, i)
            arc_w = self.adj_arcs[other][arc]
            del self.adj_arcs[other][arc]

            # add arc
            new_arc = (other, new_node_id)
            self.adj_arcs[other][new_arc] = arc_w
            self.adj_arcs[new_node_id][new_arc] = arc_w

            # add correspondance to original arcs
            if i2 < self.original_size and j2 < self.original_size:
                self.arc_to_original[new_arc] = [(i2, j2)]
            else:
                self.arc_to_original[new_arc] = self.arc_to_original[(i2, j2)]
                del self.arc_to_original[(i2, j2)]

        # same for j
        for i2, j2 in self.adj_arcs[j]:
            other = j2 if i2 == j else i2
            if other == i:
                continue
            arc = (j, other) if j < other else (other, j)
            arc_w = self.adj_arcs[other][arc]
            del self.adj_arcs[other][arc]

            new_arc = (other, new_node_id)
            # add arc
            if new_arc in self.adj_arcs[new_node_id]:
                first_w = self.adj_arcs[new_node_id][new_arc]
                if first_w > 0 and arc_w > 0:
                    arc_w = first_w + arc_w

                    if i2 < self.original_size and j2 < self.original_size:
                        self.arc_to_original[new_arc] += [(i2, j2)]
                    else:
                        self.arc_to_original[new_arc] += self.arc_to_original[(i2, j2)]
                        del self.arc_to_original[(i2, j2)]

                elif first_w > arc_w:
                    arc_w = first_w
                    # nothing to do with the mapping
                else:
                    arc_w = arc_w
                    # override the old mapping

                    if i2 < self.original_size and j2 < self.original_size:
                        self.arc_to_original[new_arc] = [(i2, j2)]
                    else:
                        self.arc_to_original[new_arc] = self.arc_to_original[(i2, j2)]
                        del self.arc_to_original[(i2, j2)]

                # update weights in the graph
                self.adj_arcs[new_node_id][new_arc] = arc_w
                self.adj_arcs[other][new_arc] = arc_w
            else:
                self.adj_arcs[other][new_arc] = arc_w
                self.adj_arcs[new_node_id][new_arc] = arc_w

                if i2 < self.original_size and j2 < self.original_size:
                    self.arc_to_original[new_arc] = [(i2, j2)]
                else:
                    self.arc_to_original[new_arc] = self.arc_to_original[(i2, j2)]
                    del self.arc_to_original[(i2, j2)]

        # delete old node and arcs
        del self.nodes[i]
        del self.nodes[j]
        del self.adj_arcs[i]
        del self.adj_arcs[j]

    def get_new_node_id(self):
        self.last_node += 1
        return self.last_node

    def get_original_from_node(self, node):
        if node < self.original_size:
            return [node], []
        else:
            return self.node_to_original_nodes[node], self.node_to_original_arcs[node]

    def get_original_from_arc(self, i, j):
        if i < self.original_size and j < self.original_size:
            return [(i, j)]
        else:
            return self.arc_to_original[(i, j)]
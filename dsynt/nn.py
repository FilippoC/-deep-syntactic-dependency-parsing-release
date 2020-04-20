import torch.nn as nn
import torch
import dsynt.algorithm

ArgmaxAlgorithms = {
    "simple": dsynt.algorithm.argmax_simple,
    "outgoing_arcs": dsynt.algorithm.argmax_outgoing_arcs,
    "semi_structured": dsynt.algorithm.argmax_semi_structured,
    "structured": dsynt.algorithm.argmax_structured
}


class ProbabilisticLoss:
    def __init__(self, reduction="sum"):
        self.builder = nn.CrossEntropyLoss(reduction=reduction)
        #self.node_builder = nn.BCEWithLogitsLoss(reduction=reduction)

    def __call__(self, weights, sentences, node_weights=None):

        n_max_words = weights.shape[1]
        n_batch = weights.shape[0]
        n_labels = weights.shape[3]

        # mask diagonals
        diagonal_mask = torch.diag(torch.empty((n_max_words,), device=weights.device).fill_(float("-inf")))
        weights = weights + diagonal_mask.reshape(1, n_max_words, n_max_words, 1)

        # add a "null label"
        null_labels = torch.zeros(weights.shape[:-1], device=weights.device, dtype=torch.float).unsqueeze(-1)
        weights = torch.cat([weights, null_labels], dim=-1)

        gold_indices = torch.empty(weights.shape[:-1], device=weights.device, dtype=torch.long, requires_grad=False).fill_(n_labels)
        for b, sentence in enumerate(sentences):
            for head, mod, label in sentence["deps"]:
                gold_indices[b, head, mod] = label

        # build mask, minus 2 to ignore BOS/EOS
        input_lengths = torch.tensor([len(sentence["words"]) - 1 for sentence in sentences], dtype=torch.long, device=weights.device)
        word_indices = torch.arange(0, n_max_words, device=input_lengths.device).unsqueeze(0).expand(n_batch, -1)
        mask = (word_indices <= input_lengths.unsqueeze(1))

        # select all mods
        # remove root from the mask
        weights = weights[mask].reshape(-1, weights.shape[-1])
        gold_indices = gold_indices[mask].reshape(-1)

        return self.builder(weights, gold_indices)


class MarginLoss:
    def __init__(self, alg, reduction="sum"):
        self.reduction = reduction
        if alg in ["semi_structured", "structured"]:
            # force the linear relaxation during training
            self.argmax = lambda a, node_weights: ArgmaxAlgorithms[alg](a, node_weights, linear_relaxation=True)
        else:
            self.argmax = ArgmaxAlgorithms[alg]

    def __call__(self, weights, sentences, node_weights=None):
        losses = []
        for i, (weight, sentence) in enumerate(zip(weights, sentences)):
            weight = weight + sentence["torch"]["deps_penalties"].to(weight.device)

            pred_arc, max_indices, pred_nodes = self.argmax(weight, node_weights=node_weights[i] if node_weights is not None else None)
            pred_arc = torch.zeros_like(weight).scatter_(2, max_indices.unsqueeze(2), pred_arc.unsqueeze(2) * 1.)

            arc_loss = (pred_arc - sentence["torch"]["deps"].to(pred_arc.device)) * weight
            if self.reduction == "sum":
                loss = torch.sum(arc_loss)
            elif self.reduction == "mean":
                loss = torch.mean(arc_loss)
            elif self.reduction == "squared_mean":
                loss = torch.sum(arc_loss) / float(weight.shape[0] * weight.shape[0])
            else:
                raise RuntimeError()

            if pred_nodes is not None:
                node_loss = (pred_nodes * 1. - sentence["torch"]["nodes"].to(pred_nodes.device)) * node_weights[i].reshape(-1)
                if self.reduction == "sum":
                    node_loss = torch.sum(node_loss)
                elif self.reduction in ["mean", "squared_mean"]:
                    node_loss = torch.mean(node_loss)
                else:
                    raise RuntimeError()

                loss = loss + node_loss
            losses.append(loss)
        return losses


class Decoder:
    def __init__(self, alg):
        self.argmax = ArgmaxAlgorithms[alg]

    def __call__(self, weight, node_weights=None):
        pred_arcs, max_indices, node_selection = self.argmax(weight, node_weights)

        pred_arcs = pred_arcs.cpu()
        max_indices = max_indices.cpu()
        arcs = list()
        n_vertices = pred_arcs.shape[0]
        for head in range(n_vertices):
            for mod in range(1, n_vertices):
                if head == mod:
                    continue
                if pred_arcs[head, mod]:
                    arcs.append((max_indices[head, mod].item(), head, mod))

        return arcs, node_selection

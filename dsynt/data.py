import pydestruct.dict
import torch
import pydestruct.nn.bert

from pydestruct.input import tensor_from_dict

def read_deep_conll(path, check_sentids=True):
    sentences = list()
    with open(path) as file:
        need_new = True
        for line in file:
            line = line.strip()
            if len(line) == 0:
                need_new = True
                continue

            line = line.split()
            if need_new:
                sentences.append({"words": list(), "tags": list(), "deps": set()})
                need_new = False

                # sentence source
                sentids = [s for s in line[5].split("|") if s.startswith("sentid=")]
                if check_sentids:
                    if len(sentids) != 1:
                        raise RuntimeError("Invalid sentence id")
                    sentences[-1]["sentid"] = sentids[0][len("sentid="):]

            sentences[-1]["words"].append(line[1])
            sentences[-1]["tags"].append(line[4])

            if line[6] == "_":
                continue
            mod = int(line[0])
            heads = [int(h) for h in line[6].split("|")]
            labels = line[7].split("|")
            for head, label in zip(heads, labels):
                # add deep dependencies only,
                # i.e. deps without type or type D:
                if not (label.startswith("S:") or label.startswith("I:")):
                    if label.startswith("D:"):
                        label = label[2:]
                    label = label.split(":")[-1]
                    sentences[-1]["deps"].add((label, head, mod))

    return sentences

def write_deep_conll(ostream, sentence, pred_deps=None):
    heads = [list() for _ in range(len(sentence["words"]))]
    labels = [list() for _ in range(len(sentence["words"]))]
    if pred_deps is None:
        pred_deps = sentence["deps"]
    for label, i, j in pred_deps:
        heads[j - 1].append(i)
        labels[j - 1].append(label)

    # 1. id
    # 2. word
    # 3. lemma
    # 4. cpos
    # 5. pos
    # 6. morph+sentid
    # 7. heads
    # 8. labels
    # 9. extra
    # 10. extra
    for i, word in enumerate(sentence["words"]):
        ostream.write(
            "%i\t%s\t_\t_\t%s\t%s\t%s\t%s\t_\t_\n"
            % (
                i + 1,
                word,
                sentence["tags"][i],
                "sentid=" + sentence["sentid"],
                "|".join(str(h) for h in heads[i]) if len(heads[i]) > 0 else "_",
                "|".join(str(h) for h in labels[i]) if len(labels[i]) > 0 else "_"
            )
        )
    ostream.write("\n")

def is_connected(sentence, details=False):
    edges = {}
    for _, i, j in sentence["deps"]:
        i, j = min(i, j), max(i, j)
        if i in edges:
            edges[i].append(j)
        else:
            edges[i] = [j]
        if j in edges:
            edges[j].append(i)
        else:
            edges[j] = [i]
    if len(edges) == 0:
        if details:
            return True, True
        else:
            return True
    visited = set()
    stack = set()
    stack.add(next(iter(edges)))
    while len(stack) > 0:
        current = stack.pop()
        visited.add(current)
        for other in edges[current]:
            if other not in visited:
                stack.add(other)

    if details:
        return len(visited) == len(edges), 0 in edges
    else:
        return len(visited) == len(edges) and 0 in edges


def build_dictionnaries(data, char_boundaries=False):
    dict_labels = set()
    dict_words = set()
    dict_chars = set()
    dict_tags = set()

    for sentence in data:
        dict_words.update(sentence["words"])
        dict_tags.update(sentence["tags"])
        for word in sentence["words"]:
            dict_chars.update(word)
        for label, _, __ in sentence["deps"]:
            dict_labels.add(label)

    dict_chars.add("**BOS_CHAR**")
    dict_chars.add("**EOS_CHAR**")
    dict_chars = pydestruct.dict.Dict(dict_chars, boundaries=char_boundaries)
    dict_tags = pydestruct.dict.Dict(dict_tags)
    dict_words = pydestruct.dict.Dict(dict_words, unk="#UNK#", boundaries=True, pad=True, lower=True)
    dict_labels = pydestruct.dict.Dict(dict_labels)

    return {"chars": dict_chars, "words": dict_words, "labels": dict_labels, "tags": dict_tags}


def build_deps_input(dicts, sentence, allow_unk_labels=False):
    deps = []
    for label, head, mod in sentence["deps"]:
        try:
            label_id = dicts["labels"].word_to_id(label)
        except KeyError as e:
            if allow_unk_labels:
                label_id = -1
            else:
                raise e

        if label_id >= 0:
            deps.append((head, mod, label_id))
    return set(deps)


def build_torch_input(sentence, dictionnaries, device="cpu", max_word_len=-1, allow_unk_labels=False, bert_tokenizer=None):
    ret = {
        "words": tensor_from_dict(dictionnaries["words"], sentence["words"], device=device),
        "chars":
            [torch.tensor([dictionnaries["chars"].word_to_id("**BOS_CHAR**")], dtype=torch.long, device=device, requires_grad=False)]
            +
            [
                tensor_from_dict(dictionnaries["chars"], t[:max_word_len] if max_word_len > 0 else t, device=device)
                for t in sentence["words"] #for word in t["form"]
            ]
            +
            [torch.tensor([dictionnaries["chars"].word_to_id("**EOS_CHAR**")], dtype=torch.long, device=device, requires_grad=False)]
        ,
        "deps": build_deps_input(dictionnaries, sentence, allow_unk_labels=allow_unk_labels),
    }

    if bert_tokenizer is not None:
        words = [pydestruct.nn.bert.BERT_TOKEN_MAPPING.get(word, word) for word in sentence["words"]]
        ret["bert"] = bert_tokenizer(words, boundaries=True, device=device)

    return ret

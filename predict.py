import argparse
import torch.optim
import sys

import pydestruct.timer
import pydestruct.data.ptb
import pydestruct.input
import pydestruct.eval
import dsynt.network
import dsynt.data
import dsynt.nn
from pydestruct.batch import batch_iterator_factory


# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--data", type=str, required=True)
cmd.add_argument("--model", type=str, required=True)
cmd.add_argument('--storage-device', type=str, default="cpu")
cmd.add_argument('--device', type=str, default="cpu")
cmd.add_argument('--export', type=str, default="")
cmd.add_argument("--batch", type=int, default=10, help="Mini-batch size")
cmd.add_argument("--batch-clusters", type=int, default=-1, help="If set, the batch is computed in number of words!")
args = cmd.parse_args()


print("Loading network from path: %s" % args.model, file=sys.stderr)
model = torch.load(args.model)
dictionnaries = model["dicts"]

network = dsynt.network.Network(
    model["args"],
    n_chars=len(dictionnaries["chars"]),
    n_words=len(dictionnaries["words"]),
    n_labels=len(dictionnaries["labels"])
)
network.to(device=args.device)
# strict should be turned to false if we decide to use pre-trained embeddings
network.load_state_dict(model["model_state_dict"], strict=True)
network.eval()


print("Reading dev data located at: %s" % args.data, file=sys.stderr)
data = dsynt.data.read_deep_conll(args.data)

print("Converting data", file=sys.stderr)
if model["args"].bert:
    bert_tokenizer = pydestruct.nn.bert.BertInputBuilder(model["args"])
else:
    bert_tokenizer = None

train_tensors = [
    dsynt.data.build_torch_input(sentence, dictionnaries, device=args.storage_device, allow_unk_labels=True, bert_tokenizer=bert_tokenizer, max_word_len=model["args"].max_word_len)
    for sentence in data
]
for i in range(len(train_tensors)):
    train_tensors[i]["index"] = i

data_iterator = batch_iterator_factory(
    train_tensors,
    args.batch,
    n_clusters=args.batch_clusters,
    size_getter=(lambda x: len(x["words"])) if args.batch_clusters > 0 else None,
    shuffle=False
)

def dev_evaluation(algorithm="simple", export=None):
    print("Evaluation with algorithm: %s" % algorithm)

    if export is not None:
        ostream = open(export, "w")

    timers = pydestruct.timer.Timers()
    decoder = dsynt.nn.Decoder("structured" if algorithm == "structured_no_node" else algorithm)
    evaluator = pydestruct.eval.ConstituentEvaluator(split_unary_chains=False)
    n_nodes, n_correct_nodes = 0, 0
    with torch.no_grad():
        for batch in data_iterator:
            timers.dev_network.start()
            batch_weights = network(batch, batched=False)
            if args.device != "cpu":
                torch.cuda.synchronize()  # ensure correct timing
            timers.dev_network.stop()

            timers.dev_parser.start()
            for b in range(len(batch)):
                weights = batch_weights[b]["deps"] + batch_weights[b]["labels"]
                node_weights = torch.zeros((weights.shape[0] - 1), device=weights.device)
                pred_deps = decoder(weights, node_weights=node_weights)

                # retrieve predicted labels
                pred = set(
                    (dictionnaries["labels"].id_to_word(label), i, j)
                    for label, i, j in pred_deps[0]
                )

                sentence = data[batch[b]["index"]]
                if export is not None:
                    dsynt.data.write_deep_conll(ostream, sentence, pred)

                evaluator.update(batch[b]["deps"], set((i, j, l) for l, i, j in pred_deps[0]))

            timers.dev_parser.stop()


    print(" - precision: %.2f" % (100 * evaluator.precision()))
    print(" - recall: %.2f" % (100 * evaluator.recall()))
    print(" - f1: %.2f" % (100 * evaluator.f1()))
    print(" - exact match: %.2f" % (100 * evaluator.exact_match()))

    print(" - network timing: %.2f sec" % timers.dev_network.seconds())
    print(" - decoder timing: %.2f sec" % timers.dev_parser.seconds())

    print()

    if export is not None:
        ostream.close()

print("Dataset: ", args.data)
dev_evaluation("simple", args.export + ".simple" if len(args.export) > 0 else None)
dev_evaluation("structured_no_node", args.export + ".structured_no_node" if len(args.export) > 0 else None)

import argparse
import sys

import pydestruct.eval
import pydestruct.logger
import pydestruct.optim
import itertools
import io
import dsynt.network
import dsynt.data
import dsynt.nn
from pydestruct.batch import batch_iterator_factory
import pydestruct.nn.bert

import transformers.optimization
import torch.optim
import pydestruct.timer
import random

def print_log(msg):
    print(msg, file=sys.stderr, flush=True)

# Read command line
cmd = argparse.ArgumentParser()
cmd.add_argument("--train", type=str, required=True, help="Path to training data")
cmd.add_argument("--dev", type=str, required=True, help="Path to dev data")
cmd.add_argument("--ensure-connected", action="store_true", help="Remove non-connected graphs")
cmd.add_argument("--model", type=str, required=True, help="Path where to store the model")
cmd.add_argument("--clip-grad", type=float, default=-1, help="grad clip value (if > 0)")
cmd.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
cmd.add_argument("--batch", type=int, default=10, help="Mini-batch size")
cmd.add_argument("--batch-clusters", type=int, default=-1, help="If set, the batch is computed in number of words!")
cmd.add_argument('--storage-device', type=str, default="cpu", help="Device where to store the data. It is useful to keep it on CPU when the dataset is large, even computation is done on GPU")
cmd.add_argument('--device', type=str, default="cpu", help="Device to use for computation")
cmd.add_argument('--char-lstm-boundaries', action="store_true", help="Add sentence boundaries at the input of the character BiLSTM")
cmd.add_argument('--tensorboard', type=str, default="", help="Tensorboard path")
cmd.add_argument('--pos-tagger', action="store_true", help="Joint POS tagger")
cmd.add_argument("--max-word-len", type=int, default=-1)
dsynt.network.Network.add_cmd_options(cmd)
pydestruct.optim.MetaOptimizer.add_cmd_options(cmd)
args = cmd.parse_args()

if len(args.tensorboard) > 0:
    print_log("Tensorboard logging: %s" % args.tensorboard)
    pydestruct.logger.open(args.tensorboard)


print_log("Reading train data located at: %s" % args.train)
train_data = dsynt.data.read_deep_conll(args.train)

print_log("Reading dev data located at: %s" % args.dev)
dev_data = dsynt.data.read_deep_conll(args.dev)

if args.ensure_connected:
    print_log("Ensuring connectedness")
    o_train_size = len(train_data)
    o_dev_size = len(dev_data)
    train_data = [sentence for sentence in train_data if dsynt.data.is_connected(sentence)]
    dev_data = [sentence for sentence in dev_data if dsynt.data.is_connected(sentence)]

    print_log("\tn. removed train sentences: %i" % (len(train_data) - o_train_size))
    print_log("\tn. removed dev sentences: %i" % (len(dev_data) - o_dev_size))

print_log("Data set size")
print_log("\tn. train sentences: %i" % len(train_data))
print_log("\tn. dev sentences: %i" % len(dev_data))

print_log("Building dictionnaries from train data")
dictionnaries = dsynt.data.build_dictionnaries(train_data, char_boundaries=args.char_lstm_boundaries)

print_log("\tn. words: %i" % len(dictionnaries["words"]))
print_log("\tn. chars: %i" % len(dictionnaries["chars"]))
print_log("\tn. labels: %i" % len(dictionnaries["labels"]))

# we build the torch object rightaway
print_log("Converting train and dev data")
if args.bert:
    bert_tokenizer = pydestruct.nn.bert.BertInputBuilder(args)
else:
    bert_tokenizer = None
train_tensors = [
    dsynt.data.build_torch_input(sentence, dictionnaries, device=args.storage_device, allow_unk_labels=False, bert_tokenizer=bert_tokenizer, max_word_len=args.max_word_len)
    for sentence in train_data
]
dev_tensors = [
    dsynt.data.build_torch_input(sentence, dictionnaries, device=args.storage_device, allow_unk_labels=True, bert_tokenizer=bert_tokenizer)
   for sentence in dev_data
]

train_data_iterator = batch_iterator_factory(
    train_tensors,
    args.batch,
    n_clusters=args.batch_clusters,
    size_getter=(lambda x: len(x["words"])) if args.batch_clusters > 0 else None,
    shuffle=True
)

dev_data_iterator = batch_iterator_factory(
    dev_tensors,
    args.batch,
    n_clusters=args.batch_clusters,
    size_getter=(lambda x: len(x["words"])) if args.batch_clusters > 0 else None,
    shuffle=False
)


print_log("Building network")
network = dsynt.network.Network(
    args,
    n_chars=len(dictionnaries["chars"]),
    n_words=len(dictionnaries["words"]),
    n_labels=len(dictionnaries["labels"])
)
network.to(device=args.device)

loss_builder = dsynt.nn.ProbabilisticLoss(reduction="sum")
decoder = dsynt.nn.Decoder("simple")

optimizer = pydestruct.optim.MetaOptimizer(args, network.parameters(), filter_freezed=True)

print_log("Training!\n")
best_dev_score = float("-inf")
best_dev_epoch = -1

print_log(
    "Epoch\tloss\tlr\t\t|"
    "\tDev (prec)\tDev (rec)\tDev (F1)\tDev (match)\tNODE prec.\tPOS prec.\tImproved?\t|"
    "\tfor.\tloss\tbackward\tepoch\t|"
    "\tforward\tparser\tdev\t\t|"
    "\telapsed"
)
timers = pydestruct.timer.Timers()
timers.total.reset(restart=True)
for epoch in range(args.epochs):
    network.train()
    epoch_loss = 0
    timers.epoch.reset(restart=True)
    timers.epoch_network.reset(restart=False)
    timers.epoch_parser.reset(restart=False)
    timers.epoch_backward.reset(restart=False)

    for batch in train_data_iterator:
        optimizer.zero_grad()

        # compute stuff
        timers.epoch_network.start()
        span_weights, label_weights = network(batch, batched=True)
        timers.epoch_network.stop()

        # compute loss
        timers.epoch_parser.start()
        batch_loss = loss_builder(
            span_weights + label_weights,
            batch
        )

        timers.epoch_parser.stop()

        # compute loss
        timers.epoch_backward.start()
        epoch_loss += batch_loss.item()

        # optimization
        batch_loss.backward()
        if args.clip_grad > 0.:
            torch.nn.utils.clip_grad_value_(network.parameters(), args.clip_grad)
        optimizer.step()
        timers.epoch_backward.stop()

    timers.epoch.stop()

    # Dev evaluation
    # We use an internal evaluator, therefore we also eval on punctuations
    # It is probably ok?
    network.eval()
    evaluator = pydestruct.eval.ConstituentEvaluator(split_unary_chains=False)
    n_tags, n_correct_tags = 0, 0
    n_nodes, n_correct_nodes = 0, 0
    timers.dev.reset(restart=True)
    timers.dev_network.reset(restart=False)
    timers.dev_parser.reset(restart=False)
    with torch.no_grad():  # no gradient is required for dev eval
        for batch in dev_data_iterator:
            timers.dev_network.start()
            batch_weights = network(batch, batched=False)

            for b in range(len(batch)):
                weights = batch_weights[b]["deps"] + batch_weights[b]["labels"]
                timers.dev_network.stop()

                timers.dev_parser.start()
                pred_deps = decoder(weights, node_weights=None)
                timers.dev_parser.stop()

                # retrieve predicted labels
                #pred = set(
                #    (dictionnaries["labels"].id_to_word(label), i, j)
                #    for label, i, j in pred_deps[0]
                #)
                # add to the evaluator
                evaluator.update(batch[b]["deps"], set((i, j, l) for l, i, j in pred_deps[0]))


    timers.dev.stop()

    # check if dev score improved
    dev_improved = False
    if evaluator.f1() > best_dev_score:
        dev_improved = True
        best_dev_score = evaluator.f1()
        best_dev_epoch = epoch
        torch.save(
            {
                "dicts": dictionnaries,
                # will be used to recreate the network, note that this can lead to privacy issue (store paths)
                # TODO: maybe use a "sub-argument pydestruct" or something like that
                "args": args,
                'model_state_dict': network.state_dict()
            },
            args.model
        )

    # print info
    # print_log(
    #"Epoch\tloss\tlr\t\t|"
    #     "\tDev (prec)\tDev (rec)\tDev (F1)\tDev (match)\tImproved?\t|"
    #     "\tforward\tloss\tbackward\tepoch\t|"
    #     "\tforward\tparser\tdev\t\t|"
    #     "\telapsed"
    # )
    print_log(
        "\r%i\t\t%.4f\t%f\t|\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%s\t\t\t|\t%.2f\t%.2f\t%.2f\t\t%.2f\t|\t%.2f\t%.2f\t%.2f\t|\t%.2f"
        %
        (
            epoch + 1,
            epoch_loss,
            optimizer.optimizer.param_groups[0]["lr"],
            100 * evaluator.precision(),
            100 * evaluator.recall(),
            100 * evaluator.f1(),
            100 * evaluator.exact_match(),
            "*" if dev_improved else " ",
            timers.epoch_network.minutes(),
            timers.epoch_parser.minutes(),
            timers.epoch_backward.minutes(),
            timers.epoch.minutes(),
            timers.dev_network.minutes(),
            timers.dev_parser.minutes(),
            timers.dev.minutes(),
            timers.total.minutes()
        )
    )

    pydestruct.logger.add_scalar("lr", optimizer.optimizer.param_groups[0]["lr"], epoch)
    pydestruct.logger.add_scalar("precision", 100 * evaluator.precision(), epoch)
    pydestruct.logger.add_scalar("recall", 100 * evaluator.recall(), epoch)
    pydestruct.logger.add_scalar("f1", 100 * evaluator.f1(), epoch)
    pydestruct.logger.add_scalar("exact_match", 100 * evaluator.exact_match(), epoch)
    pydestruct.logger.add_scalar("loss", epoch_loss, epoch)
    if n_tags > 0:
        pydestruct.logger.add_scalar("pos_precision",  100 * n_correct_tags / n_tags, epoch)

    optimizer.eval_step(evaluator.f1())

timers.total.stop()
print_log("\nDone!")
print_log("Total training time (min): %.2f" % timers.total.minutes())

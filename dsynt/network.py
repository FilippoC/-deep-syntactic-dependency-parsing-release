import torch
import torch.nn as nn

from pydestruct.network import FeatureExtractionModule
from pydestruct.nn.bilstm import BiLSTM
from pydestruct.nn.dropout import SharedDropout
from pydestruct.nn.dependency import BatchedBiaffine

class Network(nn.Module):
    def __init__(self, args, n_labels, default_lstm_init=False, **arg_dict):
        super(Network, self).__init__()

        self.feature_extractor = FeatureExtractionModule(args, default_lstm_init=default_lstm_init, **arg_dict)
        w_input_dim = self.feature_extractor.output_dim

        self.bilstm = BiLSTM(w_input_dim, args.lstm_dim, num_layers=args.lstm_layers, dropout=args.lstm_dropout)
        self.bilstm_dropout = SharedDropout(p=args.lstm_dropout)

        self.label_weights = BatchedBiaffine(
            input_dim=args.lstm_dim * 2,
            proj_dim=args.label_proj_dim,
            n_labels=n_labels,
            activation="leaky_relu",
            dropout=args.mlp_dropout,
            output_bias=False
        )
        self.span_weights = BatchedBiaffine(
            input_dim=args.lstm_dim * 2,
            proj_dim=args.span_proj_dim,
            n_labels=1,
            activation="leaky_relu",
            dropout=args.mlp_dropout,
            bias_y=True, # should be False but maybe there is a bug...
            output_bias=False
        )

    def forward(self, input, batched=False):
        features, lengths = self.feature_extractor(input)

        features = torch.nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        features, _ = self.bilstm(features)
        features, _ = torch.nn.utils.rnn.pad_packed_sequence(features, batch_first=True)

        span_weights = self.span_weights(features)
        label_weights = self.label_weights(features)

        # need to break each bach here
        if batched:
            return span_weights, label_weights
        else:
            ret = list()
            for b, l in enumerate(lengths):
                # EOS is used as root word, BOS is removed
                ret.append({"deps": span_weights[b, :l-1, :l-1], "labels": label_weights[b, :l-1, :l-1]})
            return ret

    @staticmethod
    def add_cmd_options(cmd):
        FeatureExtractionModule.add_cmd_options(cmd)

        cmd.add_argument('--biaffine', action="store_true", help="Use biaffine model")
        cmd.add_argument('--proj-dim', type=int, default=200, help="Dimension of the output projection")
        cmd.add_argument('--label-proj-dim', type=int, default=200)
        cmd.add_argument('--span-proj-dim', type=int, default=200)
        cmd.add_argument('--mlp-dropout', type=float, default=0., help="MLP dropout")

        cmd.add_argument('--lstm-dim', type=int, default=200, help="Dimension of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-layers', type=int, default=1, help="Number of layers of the sentence-level BiLSTM")
        cmd.add_argument('--lstm-dropout', type=float, default=0., help="BiLSTM dropout")

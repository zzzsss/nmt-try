import argparse

# parse the arguments for main
def init():
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    # -- data sets and dictionaries
    data.add_argument('--datasets', type=str, required=True, metavar='PATH', nargs=2,
                         help="parallel training corpus (source and target)")
    data.add_argument('--devs', type=str, required=True, metavar='PATH', nargs=2,
                         help="parallel training corpus (source and target)")
    data.add_argument('--dicts_raw', type=str, metavar='PATH', nargs="+",
                         help="raw dictionaries (one per source factor, plus target vocabulary)")
    data.add_argument('--dicts_final', type=str, metavar='PATH', nargs="+",
                         help="final dictionaries (one per source factor, plus target vocabulary), also write dest")
    data.add_argument('--rebuild_dicts', action='store_true',
                         help="rebuild dictionaries and write to files")
    data.add_argument('--dicts_thres', type=int, default=50000, metavar='INT',
                         help="cutting threshold (>100) or cutting frequency (<=100) for dicts (default: %(default)s)")
    # -- about model -- save and load
    data.add_argument('--model', type=str, default='model', metavar='PATH',
                         help="model file name (default: %(default)s)")
    data.add_argument('--reload', action='store_true',
                         help="load existing model (if '--model' points to existing model)")
    data.add_argument('--reload_model_name', type=str, metavar='PATH',
                         help="reload model file name (default: %(default)s)")
    data.add_argument('--no_reload_training_progress', action='store_false',  dest='reload_training_progress',
                         help="don't reload training progress (only used if --reload is enabled)")
    data.add_argument('--overwrite', action='store_true',
                         help="write all models to same file")

    # architecture
    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--dec_type', type=str, default="att", choices=["att", "nematus"],
                         help="decoder type (default: %(default)s)")
    network.add_argument('--att_type', type=str, default="ff", choices=["ff", "biaff"],
                         help="attention type (default: %(default)s)")
    network.add_argument('--hidden_rec', type=int, default=1000, metavar='INT',
                         help="recurrent hidden layer size (default: %(default)s)")
    network.add_argument('--hidden_att', type=int, default=1000, metavar='INT',
                         help="attention hidden layer size (default: %(default)s)")
    network.add_argument('--hidden_out', type=int, default=1000, metavar='INT',
                         help="output hidden layer size (default: %(default)s)")
    network.add_argument('--thres_src', type=int, default=None, metavar='INT',
                         help="source vocabulary threshold (default: %(default)s)")
    network.add_argument('--thres_trg', type=int, default=None, metavar='INT',
                         help="target vocabulary threshold (default: %(default)s)")
    network.add_argument('--enc_depth', type=int, default=1, metavar='INT',
                         help="number of encoder layers (default: %(default)s)")
    network.add_argument('--dec_depth', type=int, default=1, metavar='INT',         # only the first is with att
                         help="number of decoder layers (default: %(default)s)")
    network.add_argument('--factors', type=int, default=1, metavar='INT',
                         help="number of input factors (default: %(default)s)")
    network.add_argument('--dim_per_factor', type=int, default=None, nargs='+', metavar='INT',
                         help="list of word vector dimensionalities (one per factor): '--dim_per_factor 250 200 50' for total dimensionality of 500 (default: %(default)s)")
    network.add_argument('--drop_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layer (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_dec', type=float, default=0.2, metavar="FLOAT",
                         help="dropout (idrop) for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_enc', type=float, default=0.2, metavar="FLOAT",
                         help="dropout (idrop) for encoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_embedding', type=float, default=0, metavar="FLOAT",
                         help="gdrop for words (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_dec', type=float, default=0, metavar="FLOAT",
                         help="gdrop for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_enc', type=float, default=0, metavar="FLOAT",
                         help="gdrop for encoder (0: no dropout) (default: %(default)s)")

    # training progress
    training = parser.add_argument_group('training parameters')
    training.add_argument('--maxlen', type=int, default=100, metavar='INT',
                         help="maximum sequence length (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--rand_skip', type=float, default=0.0001, metavar='INT',
                         help="randomly skip batches for training (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=24, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--max_updates', type=int, default=10000000, metavar='INT',
                         help="maximum number of updates (minibatches) (default: %(default)s)")
    # -- trainer
    network.add_argument('--trainer_type', type=str, default="adam", choices=["adam", "sgd", "momentum"],
                         help="trainer type (default: %(default)s)")
    training.add_argument('--clip_c', type=float, default=1, metavar='FLOAT',
                         help="gradient clipping threshold (default: %(default)s)")
    training.add_argument('--lrate', type=float, default=0.0001, metavar='FLOAT',
                         help="learning rate or alpha (default: %(default)s)")
    training.add_argument('--moment', type=float, default=0.75, metavar='FLOAT',
                         help="momentum for mTrainer (default: %(default)s)")

    # validate
    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_freq', type=int, default=20000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    training.add_argument('--valid_batch_size', type=int, default=80, metavar='INT',
                         help="validing minibatch size (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=10, metavar='INT',
                         help="early stopping patience (default: %(default)s)")
    validation.add_argument('--anneal_restarts', type=int, default=0, metavar='INT',
                         help="when patience runs out, restart training INT times with annealed learning rate (default: %(default)s)")
    validation.add_argument('--anneal_no_renew_trainer', action='store_false',  dest='anneal_renew_trainer',
                         help="don't renew trainer (discard moments or grad info) when anneal")
    validation.add_argument('--anneal_no_reload_best', action='store_false',  dest='anneal_reload_best',
                         help="don't recovery to previous best point (discard some training) when anneal")
    validation.add_argument('--anneal_decay', type=float, default=0.5, metavar='FLOAT',
                         help="learning rate decay on each restart (default: %(default)s)")
    validation.add_argument('--validMetric', type=str, default="ll", choices=["ll", "bleu"],
                         help="type of metric for validation (default: %(default)s)")

    # common
    common = parser.add_argument_group('common')
    common.add_argument("--dynet-mem", type=str, default="")
    common.add_argument("--dynet-mem-test", action='store_true')
    common.add_argument("--dynet-autobatch", type=str, default="")
    common.add_argument("--debug", action='store_true')

    # decode (for validation or maybe certain training procedure)
    decode = parser.add_argument_group('decode')
    decode.add_argument('--beam_size', type=int, default=5,
                        help="Beam size (default: %(default)s))")
    decode.add_argument('--normalize', type=float, default=0.0, nargs="?", const=1.0, metavar="ALPHA",
                        help="Normalize scores by sentence length (with argument, exponentiate lengths by ALPHA)")

    args = parser.parse_args()
    return args

def check_network(args):
    assert args["enc_depth"] >= 1
    assert args["dec_depth"] >= 1
    assert args["factors"] >= 1
    if args["factors"] > 1:
        assert type(args["dim_per_factor"]) == list
        assert len(args["dim_per_factor"]) == args["factors"]
        assert args["dim_per_factor"][0] == args["dim_word"]    # do we need this
    if args["dim_per_factor"] is None:
        args["dim_per_factor"] = [args["dim_word"]]

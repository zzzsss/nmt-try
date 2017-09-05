import argparse
from utils import Logger

# parse the arguments for main
def init(phase):
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    if phase == "train":
        # -- data sets and dictionaries
        data.add_argument('--train', type=str, required=True, metavar='PATH', nargs=2,
                             help="parallel training corpus (source and target)")
        data.add_argument('--dev', type=str, required=True, metavar='PATH', nargs=2,
                             help="parallel dev corpus (source and target)")
        data.add_argument('--dev-output', type=str, default="dev-output.txt",
                             help="output target corpus for dev (if needed)")
        data.add_argument('--dicts_raw', type=str, metavar='PATH', nargs="+",
                             help="raw dictionaries (one per source factor, plus target vocabulary)")
        data.add_argument('--dicts_final', type=str, default=["src.v", "trg.v"], metavar='PATH', nargs="+",
                             help="final dictionaries (one per source factor, plus target vocabulary), also write dest")
        data.add_argument('--no_rebuild_dicts', action='store_false', dest='rebuild_dicts',
                             help="rebuild dictionaries and write to files")
        data.add_argument('--dicts_thres', type=int, default=50000, metavar='INT',
                             help="cutting threshold (>100) or cutting frequency (<=100) for dicts (default: %(default)s)")
        # -- about model -- save and load
        data.add_argument('--model', type=str, default='model', metavar='PATH',
                             help="model file name (default: %(default)s)")
        data.add_argument('--reload', action='store_true',
                             help="load existing model (if '--reload_model_name' points to existing model)")
        data.add_argument('--reload_model_name', type=str, metavar='PATH',
                             help="reload model file name (default: %(default)s)")
        data.add_argument('--no_reload_training_progress', action='store_false',  dest='reload_training_progress',
                             help="don't reload training progress (only used if --reload is enabled)")
        data.add_argument('--no_overwrite', action='store_false', dest='overwrite',
                             help="don't write all models to same file")
    elif phase == "test":
        # data, dictionary, model
        data.add_argument('--test', type=str, metavar='PATH', nargs=2,
                             help="parallel testing corpus (source and target)")
        data.add_argument('--output', type=str, default='output.txt', metavar='PATH', help="output target corpus")
        # data.add_argument('--gold', type=str, metavar='PATH', help="gold target corpus (for eval)") # test[1]
        data.add_argument('--dicts_final', type=str, default=["src.v", "trg.v"], metavar='PATH', nargs="+",
                          help="final dictionaries (one per source factor, plus target vocabulary), also write dest")
        data.add_argument('--models', type=str, default=["zbest.model"], metavar='PATHs', nargs="*",
                             help="model file names (ensemble if >1)")
    else:
        raise NotImplementedError(phase)

    # architecture
    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--dec_type', type=str, default="nematus", choices=["att", "nematus"],
                         help="decoder type (default: %(default)s)")
    network.add_argument('--att_type', type=str, default="ff", choices=["ff", "biaff"],
                         help="attention type (default: %(default)s)")
    network.add_argument('--hidden_rec', type=int, default=1000, metavar='INT',
                         help="recurrent hidden layer size (default: %(default)s)")
    network.add_argument('--hidden_att', type=int, default=1000, metavar='INT',
                         help="attention hidden layer size (default: %(default)s)")
    network.add_argument('--hidden_out', type=int, default=1000, metavar='INT',
                         help="output hidden layer size (default: %(default)s)")
    network.add_argument('--thres_src', type=int, default=2, metavar='INT',
                         help="source vocabulary threshold (default: %(default)s)")
    network.add_argument('--thres_trg', type=int, default=2, metavar='INT',
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
                         help="dropout for hidden layers (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_dec', type=float, default=0., metavar="FLOAT",
                         help="dropout (idrop) for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_enc', type=float, default=0., metavar="FLOAT",
                         help="dropout (idrop) for encoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_embedding', type=float, default=0., metavar="FLOAT",
                         help="gdrop for words (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_dec', type=float, default=0.2, metavar="FLOAT",
                         help="gdrop for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_enc', type=float, default=0.2, metavar="FLOAT",
                         help="gdrop for encoder (0: no dropout) (default: %(default)s)")

    # training progress
    training = parser.add_argument_group('training parameters')
    training.add_argument('--max_len', type=int, default=80, metavar='INT',
                         help="maximum sequence length (default: %(default)s)")
    training.add_argument('--fix_len_src', type=int, default=-1, metavar='INT',
                         help="fix src sentence len to this (by cutting or padding) (default: %(default)s)")
    training.add_argument('--fix_len_trg', type=int, default=-1, metavar='INT',
                         help="fix trg sentence len to this (by cutting or padding) (default: %(default)s)")
    training.add_argument('--batch_size', type=int, default=80, metavar='INT',
                         help="minibatch size (default: %(default)s)")
    training.add_argument('--rand_skip', type=float, default=0., metavar='INT',
                         help="randomly skip batches for training (default: %(default)s)")
    training.add_argument('--max_epochs', type=int, default=100, metavar='INT',
                         help="maximum number of epochs (default: %(default)s)")
    training.add_argument('--max_updates', type=int, default=1000000, metavar='INT',
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
    training.add_argument('--valid_batch_size', type=int, default=32, metavar='INT',
                         help="validing minibatch size (default: %(default)s)")
    validation.add_argument('--patience', type=int, default=5, metavar='INT',
                         help="early stopping patience (default: %(default)s)")
    validation.add_argument('--anneal_restarts', type=int, default=2, metavar='INT',
                         help="when patience runs out, restart training INT times with annealed learning rate (default: %(default)s)")
    validation.add_argument('--anneal_no_renew_trainer', action='store_false',  dest='anneal_renew_trainer',
                         help="don't renew trainer (discard moments or grad info) when anneal")
    validation.add_argument('--anneal_no_reload_best', action='store_false',  dest='anneal_reload_best',
                         help="don't recovery to previous best point (discard some training) when anneal")
    validation.add_argument('--anneal_decay', type=float, default=0.5, metavar='FLOAT',
                         help="learning rate decay on each restart (default: %(default)s)")
    validation.add_argument('--valid_metric', type=str, default="ll", choices=["ll", "bleu"],
                         help="type of metric for validation (default: %(default)s)")

    # common
    common = parser.add_argument_group('common')
    common.add_argument("--dynet-mem", type=str, default="")
    common.add_argument("--dynet-devices", type=str, default="")
    common.add_argument("--dynet-mem-test", action='store_true')
    common.add_argument("--dynet-autobatch", type=str, default="")
    common.add_argument("--dynet-seed", type=int, default=12345)    # default will be of no use, need to specify it
    common.add_argument("--debug", action='store_true')
    if phase == "train":
        common.add_argument("--log", type=str, default="z.log")
    elif phase == "test":
        common.add_argument("--log", type=str, default="")
    else:
        raise NotImplementedError(phase)

    # decode (for validation or maybe certain training procedure)
    decode = parser.add_argument_group('decode')
    decode.add_argument('--decode_type', '--decode_mode', type=str, default="decode", choices=["decode", "test1", "test2", "loop"],
                         help="type/mode of testing (decode, test, loop)")
    decode.add_argument('--decode_way', type=str, default="beam", choices=["beam", "sample"],
                         help="decoding method (default: %(default)s)")
    decode.add_argument('--beam_size', type=int, default=10,
                        help="Beam size (default: %(default)s))")
    decode.add_argument('--sample_size', type=int, default=5,
                        help="Sample size (default: %(default)s))")
    decode.add_argument('--normalize', type=float, default=0.0, metavar="ALPHA",
                        help="Normalize scores by sentence length (exponentiate lengths by ALPHA, neg means nope)")
    decode.add_argument('--decode_len', type=int, default=100, metavar='INT',
                         help="maximum decoding sequence length (default: %(default)s)")
    decode.add_argument('--decode_batched', action='store_true',
                         help="batched calculation when decoding")
    decode.add_argument('--eval_metric', type=str, default="bleu", choices=["bleu"],
                         help="type of metric for evaluation (default: %(default)s)")
    decode.add_argument('--test_batch_size', type=int, default=16, metavar='INT',
                         help="testing minibatch size (default: %(default)s)")

    a = parser.parse_args()

    # check options and some processing
    args = vars(a)
    check_options(args)
    if args["log"] is not None and len(args["log"]) > 0:    # enable logging
        Logger.start_log(args["log"])

    return args

def check_options(args):
    # network
    assert args["enc_depth"] >= 1
    assert args["dec_depth"] >= 1
    assert args["factors"] >= 1
    if args["factors"] > 1:
        assert type(args["dim_per_factor"]) == list
        assert len(args["dim_per_factor"]) == args["factors"]
        assert args["dim_per_factor"][0] == args["dim_word"]    # do we need this
    if args["dim_per_factor"] is None:
        args["dim_per_factor"] = [args["dim_word"]]

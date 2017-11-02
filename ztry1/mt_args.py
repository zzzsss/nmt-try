import argparse
import zl

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
        data.add_argument('--dev_output', type=str, default="dev-output.txt",
                             help="output target corpus for dev (if needed)")
        data.add_argument('--dicts', type=str, default=["src.v", "trg.v"], metavar='PATH', nargs="+",
                             help="final dictionaries (source / target), also write dest")
        data.add_argument('--no_rebuild_dicts', action='store_false', dest='rebuild_dicts',
                             help="rebuild dictionaries and write to files")
        data.add_argument('--dicts_rthres', type=int, default=50000, metavar='INT',
                             help="cutting threshold by rank (<= rthres) (default: %(default)s)")
        data.add_argument('--dicts_fthres', type=int, default=1, metavar='INT',
                             help="cutting threshold by freq (>= rfreq) (default: %(default)s)")
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
        data.add_argument('--test', '-t', type=str, required=True, metavar='PATH', nargs=2,
                             help="parallel testing corpus (source and target)")
        data.add_argument('--output', '-o', type=str, default='output.txt', metavar='PATH', help="output target corpus")
        # data.add_argument('--gold', type=str, metavar='PATH', help="gold target corpus (for eval)") # test[1]
        data.add_argument('--dicts', '-d', type=str, default=["src.v", "trg.v"], metavar='PATH', nargs="+",
                          help="final dictionaries (source / target)")
        data.add_argument('--models', '-m', type=str, default=["zbest.model"], metavar='PATHs', nargs="*",
                             help="model file names (ensemble if >1)")
        ## no use, just for convenience
        data.add_argument('--dicts_rthres', type=int, default=50000, metavar='INT', help="NON-USED OPTION")
        data.add_argument('--dicts_fthres', type=int, default=1, metavar='INT', help="NON-USED OPTION")
    else:
        raise NotImplementedError(phase)

    # architecture
    network = parser.add_argument_group('network parameters')
    network.add_argument('--dim_word', type=int, default=512, metavar='INT',
                         help="embedding layer size (default: %(default)s)")
    network.add_argument('--dec_type', type=str, default="nematus", choices=["att", "nematus"],
                         help="decoder type (default: %(default)s)")
    network.add_argument('--att_type', type=str, default="ff", choices=["ff", "biaff", "dummy"],
                         help="attention type (default: %(default)s)")
    network.add_argument('--rnn_type', type=str, default="gru", choices=["gru", "gru2", "lstm", "dummy"],
                         help="recurrent node type (default: %(default)s)")
    network.add_argument('--summ_type', type=str, default="ends", choices=["avg", "fend", "bend", "ends"],
                         help="decoder's starting summarizing type (default: %(default)s)")
    network.add_argument('--hidden_rec', type=int, default=1000, metavar='INT',
                         help="recurrent hidden layer (default for dec&&enc) (default: %(default)s)")
    network.add_argument('--hidden_dec', type=int, metavar='INT',
                         help="decoder hidden layer size (default: hidden_rec")
    network.add_argument('--hidden_enc', type=int, metavar='INT',
                         help="encoder hidden layer size <BiRNN thus x2> (default: hidden_rec")
    network.add_argument('--hidden_att', type=int, default=1000, metavar='INT',
                         help="attention hidden layer size (default: %(default)s)")
    network.add_argument('--hidden_out', type=int, default=500, metavar='INT',
                         help="output hidden layer size (default: %(default)s)")
    network.add_argument('--dim_cov', type=int, default=0, metavar='INT',
                         help="dimension for coverage in att (default: %(default)s)")
    network.add_argument('--enc_depth', type=int, default=1, metavar='INT',
                         help="number of encoder layers (default: %(default)s)")
    network.add_argument('--dec_depth', type=int, default=1, metavar='INT',         # only the first is with att
                         help="number of decoder layers (default: %(default)s)")
    network.add_argument('--drop_hidden', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for hidden layers (0: no dropout) (default: %(default)s)")
    network.add_argument('--drop_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="dropout for embeddings (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_embedding', type=float, default=0.2, metavar="FLOAT",
                         help="idrop for words (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_embedding', type=float, default=0., metavar="FLOAT",
                         help="gdrop for words (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_rec', type=float, default=0., metavar="FLOAT",
                         help="dropout (idrop) for recurrent nodes (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_dec', type=float, metavar="FLOAT",
                         help="dropout (idrop) for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--idrop_enc', type=float, metavar="FLOAT",
                         help="dropout (idrop) for encoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_rec', type=float, default=0., metavar="FLOAT",
                         help="gdrop for recurrent nodes (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_dec', type=float, metavar="FLOAT",
                         help="gdrop for decoder (0: no dropout) (default: %(default)s)")
    network.add_argument('--gdrop_enc', type=float, metavar="FLOAT",
                         help="gdrop for encoder (0: no dropout) (default: %(default)s)")

    # training progress
    training = parser.add_argument_group('training parameters')
    training.add_argument('--no_shuffle_training_data', action='store_false', dest='shuffle_training_data',
                             help="don't shuffle training data before each epoch")
    network.add_argument('--training_sort_type', type=str, default="trg-src", choices=["src", "trg", "src-trg", "trg-src"],
                         help="training data's sort type (default: %(default)s)")
    training.add_argument('--max_len', type=int, default=50, metavar='INT',
                         help="maximum sequence length (default: %(default)s)")
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
    training.add_argument('--moment', type=float, default=0.8, metavar='FLOAT',
                         help="momentum for mTrainer (default: %(default)s)")

    # -- validate
    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_freq', type=int, default=10000, metavar='INT',
                         help="validation frequency (default: %(default)s)")
    training.add_argument('--valid_batch_size', '--valid_batch_width', type=int, default=8, metavar='INT',
                         help="validating minibatch-size (default: %(default)s)")
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
    validation.add_argument('--valid_metrics', type=str, default="bleu,ll",
                         help="type of metric for validation (separated by ',') (default: %(default)s)")
    validation.add_argument('--validate_epoch', action='store_true',
                             help="validate at the end of each epoch")

    # common
    common = parser.add_argument_group('common')
    common.add_argument("--dynet-mem", type=str, default="4", dest="dynet-mem")
    common.add_argument("--dynet-devices", type=str, default="CPU", dest="dynet-devices")
    common.add_argument("--dynet-autobatch", type=str, default="0", dest="dynet-autobatch")
    common.add_argument("--dynet-seed", type=str, default="12345", dest="dynet-seed")    # default will be of no use, need to specify it
    common.add_argument("--dynet-immed", action='store_true', dest="dynet-immed")
    common.add_argument("--bk_init_nl", type=str, default="glorot", )
    common.add_argument("--debug", action='store_true')
    common.add_argument("--verbose", "-v", action='store_true')
    common.add_argument("--log", type=str, default=zl.utils.Logger.MAGIC_CODE, help="logger for the process")
    common.add_argument('--report_freq', type=int, default=800, metavar='INT',
                         help="report frequency (number of instances / only when verbose) (default: %(default)s)")

    # decode (for validation or maybe certain training procedure)
    decode = parser.add_argument_group('decode')
    # decode.add_argument('--decode_type', '--decode_mode', type=str, default="decode", choices=["decode", "decode_gold", "test1", "test2", "loop"],
    #                      help="type/mode of testing (decode, test, loop)")
    # decode.add_argument('--decode_way', type=str, default="beam", choices=["beam", "sample"],
    #                      help="decoding method (default: %(default)s)")
    decode.add_argument('--beam_size', '-k', type=int, default=10, help="Beam size (default: %(default)s))")
    # todo: additive, gaussian
    decode.add_argument('--normalize', '-n', type=float, default=0.0, metavar="ALPHA",
                         help="Normalize scores by sentence length (exponentiate lengths by ALPHA, neg means nope)")
    decode.add_argument('--normalize_way', type=str, default="norm", choices=["norm", "google", "gaussian", "xgaussian"],
                         help="how to norm length (default: %(default)s)")
    decode.add_argument('--decode_len', type=int, default=80, metavar='INT',
                         help="maximum decoding sequence length (default: %(default)s)")
    # decode.add_argument('--decode_ratio', type=float, default=10.,
    #                      help="maximum decoding sequence length ratio compared with src (default: %(default)s)")
    decode.add_argument('--eval_metric', type=str, default="bleu", choices=["bleu", "nist"],
                         help="type of metric for evaluation (default: %(default)s)")
    decode.add_argument('--test_batch_size', type=int, default=8, metavar='INT',
                         help="testing minibatch-size(default: %(default)s)")

    # extra: for training
    training2 = parser.add_argument_group('training parameters section2')
    # scale original loss function for training
    training2.add_argument('--train_scale', type=float, default=0.0, metavar="ALPHA",
                         help="(train2) Scale scores by sentence length (exponentiate lengths by ALPHA, neg means nope)")
    training2.add_argument('--train_scale_way', type=str, default="norm", choices=["norm", "google"],
                         help="(train2) how to norm length with score scales (default: %(default)s)")
    # length fitting for training


    a = parser.parse_args()

    # check options and some processing
    args = vars(a)
    check_options(args)
    # init them
    zl.init_all(args)
    return args

def check_options(args):
    # network
    assert args["enc_depth"] >= 1
    assert args["dec_depth"] >= 1
    # defaults
    for prefix in ["hidden_", "idrop_", "gdrop_"]:
        for n in ["dec", "enc"]:
            n0, n1 = prefix+n, prefix+"rec"
            if args[n0] is None:
                args[n0] = args[n1]
    # validation
    VALID_OPTIONS = ["ll", "bleu"]
    s = args["valid_metrics"].split(",")
    assert all([one in VALID_OPTIONS for one in s])
    args["valid_metrics"] = s

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Sequence arguments
    parser.add_argument('--snr1', type=float, default= -1, help="Transmission SNR")
    parser.add_argument('--snr2', type=float, default= 100., help="Feedback SNR")
    parser.add_argument('--K', type=int, default=51, help="Sequence length")
    parser.add_argument('--m', type=int, default=3, help="Block size")
    parser.add_argument('--ell', type=int, default=17, help="Number of bit blocks")
    parser.add_argument('--T', type=int, default=9, help="Number of interactions")
    parser.add_argument('--seq_reloc', type=int, default=1)
    parser.add_argument('--memory', type=int, default=51)
    parser.add_argument('--core', type=int, default=1)
    parser.add_argument('--NS_model', type=int, default=2)

    # Transformer arguments
    parser.add_argument('--heads_trx', type=int, default=1, help="number of heads for the multi-head attention")
    parser.add_argument('--d_k_trx', type=int, default=32, help="number of features for each head")
    parser.add_argument('--N_trx', type=int, default=2, help=" number of layers in the encoder and decoder")
    parser.add_argument('--dropout', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--custom_attn', type=bool, default = True, help= "use custom attention")
    parser.add_argument('--vv', type=int, default=1)

    # Learning arguments
    parser.add_argument('--load_weights') # None
    parser.add_argument('--train', type=int, default= 0)
    parser.add_argument('--reloc', type=int, default=1, help="w/ or w/o power rellocation")
    parser.add_argument('--totalbatch', type=int, default=120101, help="number of total batches to train")
    parser.add_argument('--batchSize', type=int, default=8192, help="batch size")
    parser.add_argument('--opt_method', type=str, default='adamW', help="Optimization method adamW,lamb,adam")
    parser.add_argument('--clip_th', type=float, default=0.5, help="clipping threshold")
    parser.add_argument('--use_lr_schedule', type=bool, default = True, help="lr scheduling")
    parser.add_argument('--multclass', type=bool, default = True, help="bit-wise or class-wise training")
    parser.add_argument('--embedding', type=bool, default = False, help="vector embedding option")
    parser.add_argument('--embed_normalize', type=bool, default = True, help="normalize embedding")
    parser.add_argument('--belief_modulate', type=bool, default = True, help="modulate belief [-1,1]")
    parser.add_argument('--clas', type=int, default = 8, help="number of possible class for a block of bits")
    parser.add_argument('--rev_iter', type=int, default = 0, help="number of successive iteration")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('--device', type=str, default='cuda:0', help="GPU")
    args = parser.parse_args()

    return args

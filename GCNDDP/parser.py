import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')  # 256
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size')  # 4096
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=1e-3, type=float, help='weight of cl loss')  # 0.05
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--d', default=1024, type=int, help='embedding size')#512 0.886
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=4, type=int, help='number of gnn layers')  # 2
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.1, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.8, type=float, help='temperature in cl loss')  # 0.8
    parser.add_argument('--lambda2', default=1e-5, type=float, help='l2 reg weight')  # 1e-5
    parser.add_argument('--cuda', default='1', type=str, help='the gpu to use')
    return parser.parse_args()
args = parse_args()
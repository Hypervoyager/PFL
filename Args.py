import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='fed_avg',
                        help='Type of algorithms:{fed_mutual, fed_avg, fed_coteaching, normal, parallel}')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument('--node_num', type=int, default=10,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=200,
                        help='Number of rounds: R')
    parser.add_argument('--E', type=int, default=5,
                        help='Number of local epochs: E')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes of Experiments')
    parser.add_argument('--max_lost', type=int, default=1,
                        help='The difference in the number of communication rounds between the fastest and slowest nodes ')
    parser.add_argument('--warmup', type=int, default=5,
                        help='The number of warmup')
    parser.add_argument('--mu', type=float, default=0.2,
                        help='Degree of non-iid')
                        

    # Model
    parser.add_argument('--global_model', type=str, default='ResNet18',
                        help='Type of global model: {LeNet5, MLP, CNN2, ResNet18}')
    parser.add_argument('--local_model', type=str, default='ResNet18',
                        help='Type of local model: {LeNet5, MLP, CNN2, ResNet18}')

    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='datasets: {cifar100, cifar10, femnist, mnist}')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='batchsize')
    parser.add_argument('--split', type=int, default=5,
                        help='data split')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='val_ratio')
    parser.add_argument('--all_data', type=bool, default=True,
                        help='use all train_set')
    parser.add_argument('--classes', type=int, default=10,
                        help='classes')
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")
    parser.add_argument('--sampler', type=str, default='iid', help="iid, non-iid")

    # noise
    parser.add_argument('--noise_rate', type=float, default=0,
                        help='噪声比例')
    parser.add_argument('--noise_type', type=str, default='clean',
                        help='噪声类型： {symmetric， asymmetric, clean}')
    parser.add_argument('--num_gradual', type=int, default=5,
                        help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to '
                             'Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type=float, default=1,
                        help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in '
                             'Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--loss', type=str, default='CE',
                        help='loss:{CE, GCE}')
    parser.add_argument('--is_sparse', type=int, default=0,
                        help='if use the sparse regularizatoin mechanism')


    # Optima
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer: {sgd, adam}')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='learning rate decay step size')
    parser.add_argument('--stop_decay', type=int, default=50,
                        help='round when learning rate stop decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='local ratio of data loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='global ratio of data loss')

    args = parser.parse_args()
    return args

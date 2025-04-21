import argparse
from .auto_naming import get_exp_name

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument("--arch", default="resnet20", choices=['resnet20', 'resnet18', 'resnet50'],
        help="model architecture")
    parser.add_argument('--data_dir', default='~/data')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100','tinyimagenet'],
                        help='dataset: ' + ' (default: cifar10)')
    parser.add_argument("--num_workers",default=4, type=int,
        help="number of data loading workers (default: 4)")
    parser.add_argument("--epochs", default=200, type=int, metavar="N", 
        help="number of total epochs to run")
    parser.add_argument("--resume_from_epoch", default=0, type=int,
        help="resume from a specific epoch")
    parser.add_argument("--batch_size", default=128, type=int,
        help="mini-batch size (default: 128)")
    parser.add_argument("--lr", default=0.1, type=float,help="initial learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float,
        help="weight decay (default: 5e-4)")
    parser.add_argument("--save-dir", default="./outputs", type=str,
        help="The directory used to save output")
    parser.add_argument("--save_freq", type=int, default=200,
        help="Saves checkpoints at every specified number of epochs")
    parser.add_argument("--gpu", type=int, nargs='+', default=[0])

    parser.add_argument("--trainer_type", default="crest", choices=['none', 'random', 'crest'],
        help="Type of trainer to use (none=base, random=random selection, crest=core set selection)")
    parser.add_argument("--smtk", type=int, help="smtk", default=0)
    parser.add_argument("--train_frac", "-s", type=float, default=0.1, help="training fraction")
    parser.add_argument("--lr_milestones", type=int, nargs='+', default=[100,150])
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate decay parameter")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument("--runs", type=int, help="num runs", default=1)
    parser.add_argument("--warm_start_epochs", default=20, type=int, help="epochs to warm start learning rate")
    parser.add_argument("--subset_start_epoch", default=0, type=int, help="epoch to start subset selection")

    # data augmentation options
    parser.add_argument("--cache_dataset", default=True, type=parse_bool, const=True, nargs='?',
        help="cache the dataset in memory")
    parser.add_argument("--clean_cache_selection", default=False, type=parse_bool, const=True, nargs='?',
        help="clean the cache when selecting a new subset")
    parser.add_argument("--clean_cache_iteration", default=True, type=parse_bool, const=True, nargs='?',
        help="clean the cache after iterating over the dataset")

    # Crest options
    parser.add_argument("--approx_moment", default=True, type=parse_bool, const=True, nargs='?',
        help="use momentum in approximation")
    parser.add_argument("--approx_with_coreset", default=True, type=parse_bool, const=True, nargs='?',
        help="use all (selected) coreset data for loss function approximation")
    parser.add_argument("--check_interval", default=1, type=int,
        help="frequency to check the loss difference")
    parser.add_argument("--num_minibatch_coreset", default=5, type=int, 
        help="number of minibatches to select together")
    parser.add_argument("--batch_num_mul", default=5, type=float, 
        help="multiply the number of minibatches to select together")
    parser.add_argument("--interval_mul", default=1., type=float, 
        help="multiply the interval to check the loss difference")
    parser.add_argument("--check_thresh_factor", default=0.1,type=float,
        help="use loss times this factor as the loss threshold",)
    parser.add_argument("--shuffle", default=True, type=parse_bool, const=True, nargs='?',
        help="use shuffled minibatch coreset")

    # random subset options
    parser.add_argument("--random_subset_size", default=0.01, type=float, 
        help="partition the training data to select subsets")
    parser.add_argument("--partition_start", default=0, type=int, 
        help="which epoch to start selecting by minibatches")

    # dropping examples below a loss threshold
    parser.add_argument('--drop_learned', default=False, type=parse_bool, const=True, nargs='?', help='drop learned examples')
    parser.add_argument('--watch_interval', default=5, type=int, help='decide whether an example is learned based on how many epochs')
    parser.add_argument('--drop_interval', default=20, type=int, help='decide whether an example is learned based on how many epochs')
    parser.add_argument('--drop_thresh', default=0.1, type=float, help='loss threshold')
    parser.add_argument('--min_train_size', default=40000, type=int)
    parser.add_argument('--min_batch_size', default=400, type=int)
    parser.add_argument('--generate_mixed_subset', default=True, type=bool, help='whether to generate a mixed subset')
    # detrimental example dropping
    parser.add_argument('--drop_detrimental', default=True, type=parse_bool, const=True, nargs='?', help='drop detrimental examples')
    parser.add_argument('--cluster_thresh', default=1, type=int, help='cluster size threshold')
    parser.add_argument('--detrimental_cluster_num', default=64, type=int, help='Number of clusters to create.')
    parser.add_argument('--target_drop_percentage', default=0, type=int, help='Further drop after cluster_thresh drop.')
    parser.add_argument('--drop_after', default=0, type=int, help='epoch to start dropping detrimental examples')
    parser.add_argument('--optimizer', default="LazyGreedy", type=str, help='optimizer for detrimental instance dropping')
    
    # DPP adaptive weight parameters
    parser.add_argument('--adaptive_dpp', default=False, type=parse_bool, const=True, nargs='?', help='Use adaptive DPP weights')
    parser.add_argument('--use_learnable_lambda', default=True, type=parse_bool, const=True, nargs='?', help='Use truly learnable lambda parameter')
    parser.add_argument('--dpp_weight', default=0.5, type=float, help='Base DPP weight when not using adaptive/learnable weighting')
    parser.add_argument('--min_dpp_weight', default=0.2, type=float, help='Minimum DPP weight for adaptive/learnable weighting')
    parser.add_argument('--max_dpp_weight', default=0.8, type=float, help='Maximum DPP weight for adaptive/learnable weighting')
    parser.add_argument('--dpp_schedule_factor', default=1.0, type=float, help='How quickly to increase diversity weight during training')
    parser.add_argument('--gradient_alignment_threshold', default=0.7, type=float, help='Threshold for gradient similarity that triggers diversity boost')
    parser.add_argument('--meta_lr', default=0.01, type=float, help='Learning rate for meta-optimization of lambda')
    parser.add_argument('--per_class_lambda', default=True, type=parse_bool, const=True, nargs='?', help='Use per-class lambda parameters')
    
    # Spectral influence selection parameters (alternative to DPP)
    parser.add_argument('--selection_method', default="mixed", type=str, choices=['mixed', 'dpp', 'submod', 'spectral', 'trimodal', 'rand'], 
                       help='Method to use for subset selection (mixed=DPP+coverage, spectral=non-DPP alternative, trimodal=combines all three approaches)')
    parser.add_argument('--n_clusters_factor', default=0.1, type=float, 
                       help='Factor to determine number of clusters as fraction of subset size (for spectral method)')
    parser.add_argument('--influence_type', default="gradient_norm", type=str, 
                       choices=['gradient_norm', 'loss', 'uncertainty'], 
                       help='Method to compute influence scores (for spectral method)')
    parser.add_argument('--balance_clusters', default=True, type=parse_bool, const=True, nargs='?',
                       help='Whether to enforce balanced selection across clusters (for spectral method)')
    parser.add_argument('--affinity_metric', default="rbf", type=str, choices=['rbf', 'cosine'], 
                       help='Metric for computing the affinity matrix (for spectral method)')
    
    # Trimodal weight parameters
    parser.add_argument('--spectral_weight', default=0.33, type=float,
                      help='Weight allocated to spectral influence selection in trimodal method (0-1)')
    parser.add_argument('--dpp_submod_ratio', default=0.5, type=float,
                      help='Ratio between DPP and Submodular in remaining budget after spectral (0=all submod, 1=all DPP)')

    # others
    parser.add_argument('--use_wandb', default=False, type=parse_bool, const=True, nargs='?')

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
    else:
        raise NotImplementedError
    
    args.save_dir = get_exp_name(args)
    
    return args


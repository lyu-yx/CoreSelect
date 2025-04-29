from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import submodular, craig
import os
import time

# Global cache to store selected subsets across epochs
# Maps (selection_method, subset_size, dataset_size) -> (subset_indices, weights, timestamp)
_SUBSET_CACHE = {}

# Import the cache from the other SubsetGenerator implementation
try:
    from datasets.subset_generator import GLOBAL_SUBSET_CACHE
except ImportError:
    # If import fails, create our own cache
    GLOBAL_SUBSET_CACHE = {}

def distribute_subset(subset, weight, ordering_time, similarity_time, pred_time, args):
    size = torch.Tensor([len(subset)]).int().cuda()
    subset_sizes = [
        torch.zeros(size.shape, dtype=torch.int32).cuda()
        for _ in range(args.world_size)
    ]
    dist.all_gather(subset_sizes, size)
    max_size = torch.max(torch.cat(subset_sizes)).item()

    subset_list = [
        torch.zeros(max_size, dtype=torch.int64).cuda() for _ in range(args.world_size)
    ]

    subset = (
        np.append(subset, [0] * (max_size - len(subset)))
        if len(subset) != max_size
        else subset
    )

    dist.all_gather(subset_list, torch.from_numpy(subset).cuda())
    subset_list = [
        subset_list[i][: subset_sizes[i].item()] for i in range(args.world_size)
    ]
    subset = torch.cat(subset_list).cpu().numpy()

    if args.weighted and args.greedy:
        weight_list = [
            torch.zeros(max_size, dtype=torch.float32).cuda()
            for _ in range(args.world_size)
        ]

        weight = (
            torch.cat(
                [
                    weight,
                    torch.zeros(max_size - len(weight), dtype=torch.float32).cuda(),
                ]
            )
            if len(weight) != max_size
            else weight
        )

        dist.all_gather(weight_list, weight)
        weight_list = [
            weight_list[i][: subset_sizes[i].item()] for i in range(args.world_size)
        ]
        weight = torch.cat(weight_list)

    reduced_times = (
        torch.Tensor([ordering_time, similarity_time, pred_time]).float().cuda()
    )
    dist.reduce(
        reduced_times, 0, op=dist.ReduceOp.MAX,
    )

    return subset, weight, reduced_times


class SubsetMode(Enum):
    GREEDY = 1
    RANDOM = 2
    CLUSTER = 3


def cluster_features(train_dir, train_num, normalize):
    data = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    preds, labels = (
        np.reshape(data.imgs, (train_num, -1)),
        data.targets,
    )

    return preds, labels


class SubsetGenerator:
    def __init__(self, greedy, smtk):
        self.mode = self._get_mode(greedy)
        self.smtk = smtk
        
        # Add cache control parameters
        self.use_cache = True  # Default to using cache
        self.cache_lifetime = 10  # Default cache lifetime (in epochs)
        self.last_full_selection_epoch = -1
        self.current_epoch = -1
        self.refresh_frequency = 10
        
        # Control verbose output
        self.verbose = os.environ.get('VERBOSE_SELECTION', '0') == '1'

    def set_epoch(self, epoch: int):
        """
        Set the current epoch for the generator
        This is critical for controlling refresh frequency
        """
        self.current_epoch = epoch
        
    def set_refresh_frequency(self, freq: int):
        """
        Set how often to refresh selection (every N epochs)
        """
        self.refresh_frequency = freq
        
    def force_selection_refresh(self):
        """
        Force a selection refresh on the next call
        """
        self.force_refresh = True
        
    def _should_refresh(self, epoch: int) -> bool:
        """
        Determine if we should refresh the selection based on:
        1. If forced refresh is requested
        2. If this is a new epoch (different from last cached epoch)
        3. If we've reached the refresh frequency interval
        """
        if hasattr(self, 'force_refresh') and self.force_refresh:
            self.force_refresh = False  # Reset flag
            return True
            
        # If first selection or epoch changed since last time
        if self.current_epoch < 0 or epoch != self.current_epoch:
            self.current_epoch = epoch
            
            # Only refresh every N epochs
            needs_refresh = epoch % self.refresh_frequency == 0
            return needs_refresh
            
        # Default: no refresh needed
        return False

    def _get_mode(self, greedy):
        if not greedy:
            return SubsetMode.RANDOM
        else:
            return SubsetMode.GREEDY

    def _random_subset(self, B, idx):
        # order = np.arange(0, TRAIN_NUM)
        # order = idx  # todo
        # np.random.shuffle(order)  # todo: with replacement
        # subset, weight = order[:B], None
        rnd_idx = np.random.randint(0, len(idx), B)
        subset, weight = idx[rnd_idx], None

        return subset, weight, 0, 0, 0

    def _greedy_features(
        self,
        epoch,
        train_dataset,
        idx,
        batch_size,
        workers,
        class_num,
        targets,
        predict_function,
        predict_args,
        pred_loader=None,
    ):
        if pred_loader is None:
            idx_subset = torch.utils.data.Subset(train_dataset, indices=idx)
            pred_loader = DataLoader(
                idx_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
            )  # (Note): shuffle=False

        preds, pred_time = predict_function(pred_loader, *predict_args)
        preds = preds[idx]
        
        # Only print if verbose
        if self.verbose:
            print(f"Epoch [{epoch}] [Greedy], pred size: {np.shape(preds)}")
            
        if np.shape(preds)[-1] == class_num:
            preds -= np.eye(class_num)[targets[idx]]

        return preds, pred_time

    def generate_subset(
        self,
        preds,
        epoch,
        B,
        idx,
        targets,
        subset_printer=None,
        mode="dense",
        num_n=128,
        use_submodlib=True,
        use_cache=True,
        force_new_selection=False
    ):
        """
        Generate a subset with efficient caching to avoid redundant computation
        """
        # Update current epoch
        self.current_epoch = epoch
        
        # Create a cache key based on selection parameters
        data_hash = hash(
            str(preds.shape) + str(np.mean(preds)) + 
            str(len(idx)) + str(hash(str(targets)))
        )
        cache_key = f"{str(self.mode)}_{B}_{len(idx)}_{data_hash}"
        unified_cache_key = f"subset_{B}_{len(idx)}_{data_hash}"
        
        # Check GLOBAL_SUBSET_CACHE first (shared with subset_generator.py)
        if not force_new_selection and use_cache and unified_cache_key in GLOBAL_SUBSET_CACHE:
            cached_subset, cached_weight, cached_epoch = GLOBAL_SUBSET_CACHE[unified_cache_key]
            
            # Only use cache if we don't need a refresh
            if not self._should_refresh(epoch):
                if self.verbose:
                    print(f"[UNIFIED CACHE] Reusing subset from epoch {cached_epoch} (current epoch {epoch})")
                return cached_subset, cached_weight, 0.0, 0.0
                
        # Then check local cache
        if not force_new_selection and use_cache and self.use_cache:
            if cache_key in _SUBSET_CACHE:
                cached_subset, cached_weight, cached_epoch = _SUBSET_CACHE[cache_key]
                
                # Check if the cache is still valid (within cache_lifetime)
                if not self._should_refresh(epoch):
                    # Use cached result with minimal computation
                    if self.verbose:
                        print(f"[LOCAL CACHE] Using cached subset from epoch {cached_epoch} (saving computation)")
                    
                    # Also update the unified cache for sharing
                    GLOBAL_SUBSET_CACHE[unified_cache_key] = (cached_subset, cached_weight, epoch)
                    return cached_subset, cached_weight, 0.0, 0.0
        
        # If not using cache or cache miss or forced refresh, perform full selection
        if self.verbose:
            print(f"[SELECTION] Performing full subset selection at epoch {epoch}")
            
        if subset_printer is not None:
            subset_printer.print_selection(self.mode, epoch)

        # Record this as the last full selection epoch
        self.last_full_selection_epoch = epoch
        
        # Minimal logging to reduce output spam
        if self.verbose:
            print(f"Epoch [{epoch}] [{'Greedy' if self.mode == SubsetMode.GREEDY else 'Random'}], initial pred shape: {preds.shape}")

        fl_labels = targets[idx] - np.min(targets[idx])
        start_time = time.time()

        if len(fl_labels) > 50000:
            (
                subset,
                weight,
                _,
                _,
                ordering_time,
                similarity_time,
            ) = submodular.greedy_merge(preds, fl_labels, B, 5, "euclidean",)
        else:
            if use_submodlib:
                (
                    subset,
                    weight,
                    _,
                    _,
                    ordering_time,
                    similarity_time,
                ) = submodular.get_orders_and_weights(
                    B,
                    preds,
                    "euclidean",
                    y=fl_labels,
                    equal_num=True,
                    mode=mode,
                    num_n=num_n,
                )
            else:
                (
                    subset,
                    weight,
                    _,
                    _,
                    ordering_time,
                    similarity_time,
                ) = craig.get_orders_and_weights(
                    B, preds, "euclidean", y=fl_labels, equal_num=True, smtk=self.smtk
                )

            subset = np.array(idx[subset])  # (Note): idx
            weight = np.array(weight)
            
        # Store in both caches for future use
        if use_cache:
            # Local cache
            _SUBSET_CACHE[cache_key] = (subset, weight, epoch)
            
            # Unified cache (shared with subset_generator.py)
            GLOBAL_SUBSET_CACHE[unified_cache_key] = (subset, weight, epoch)
            
        # Only log selection time if verbose
        if self.verbose:
            total_time = time.time() - start_time
            print(f"Selection completed in {total_time:.2f} seconds, selected {len(subset)} points")

        return subset, weight, ordering_time, similarity_time


class WeightedSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        target = self.dataset[self.indices[idx]][1]
        return image, target, self.weights[idx]

    def __len__(self):
        return len(self.indices)


def get_coreset(gradient_est, 
                labels, 
                N, 
                B, 
                num_classes, 
                equal_num=True,
                optimizer="LazyGreedy",
                metric='euclidean',
                epoch=-1):  # Added epoch parameter
    '''
    Arguments:
        gradient_est: Gradient estimate
            numpy array - (N,p) 
        labels: labels of corresponding grad ests
            numpy array - (N,)
        B: subset size to select
            int
        num_classes:
            int
        normalize_weights: Whether to normalize coreset weights based on N and B
            bool
        gamma_coreset:
            float
        smtk:
            bool
        st_grd:
            bool
        epoch: Current training epoch (-1 means unknown)
            int

    Returns 
    (1) coreset indices (2) coreset weights (3) ordering time (4) similarity time
    '''
    # Try to use cached results if available (added epoch-aware caching)
    data_hash = hash(str(gradient_est.shape) + str(np.mean(gradient_est)) + str(hash(str(labels))))
    cache_key = f"coreset_{B}_{N}_{data_hash}"
    
    if epoch >= 0 and cache_key in GLOBAL_SUBSET_CACHE:
        cached_subset, cached_weights, cached_epoch = GLOBAL_SUBSET_CACHE[cache_key]
        
        # Only use cache if we're not at a refresh epoch
        # Use modulo 10 as default refresh frequency
        if epoch % 10 != 0 and epoch != cached_epoch:
            print(f"[CACHE] Reusing coreset from epoch {cached_epoch}")
            return cached_subset, cached_weights, 0.0, 0.0, None
    
    # Perform actual selection if cache miss or refresh needed
    try:
        subset, subset_weights, _, _, ordering_time, similarity_time, cluster = submodular.get_orders_and_weights_detrimental(
            B, 
            gradient_est, 
            metric, 
            y=labels, 
            equal_num=equal_num, 
            num_classes=num_classes,
            optimizer=optimizer)
    except ValueError as e:
        print(e)
        print(f"WARNING: ValueError from coreset selection, choosing random subset for this epoch")
        subset, subset_weights = get_random_subset(B, N)
        ordering_time = 0
        similarity_time = 0
        cluster = None

    if len(subset) != B:
        print(f"!!WARNING!! Selected subset of size {len(subset)} instead of {B}")
    print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')
    
    # Cache the result if epoch is provided
    if epoch >= 0:
        GLOBAL_SUBSET_CACHE[cache_key] = (subset, subset_weights, epoch)

    return subset, subset_weights, ordering_time, similarity_time, cluster


def get_random_subset(B, N):
    print(f'Selecting {B} element from the random subset of size: {N}')
    order = np.arange(0, N)
    np.random.shuffle(order)
    subset = order[:B]

    return subset



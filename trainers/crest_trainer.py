from utils import Adahessian
from utils.learnable_lambda import LearnableLambda
from datasets.subset import get_coreset
from .subset_trainer import *
import time
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


class CRESTTrainer(SubsetTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        self.train_indices = np.arange(len(self.train_dataset))
        self.steps_per_epoch = np.ceil(int(len(self.train_dataset) * self.args.train_frac) / self.args.batch_size).astype(int)
        self.reset_step = self.steps_per_epoch

        # Set default log_interval if not provided
        if not hasattr(self.args, 'log_interval'):
            self.args.log_interval = 10  # Log every 10 batches by default

        # Performance metrics
        self.approx_time = AverageMeter()
        self.selection_time = AverageMeter()
        self.similarity_time = AverageMeter()
        
        # Gradient approximation
        self.gradient_approx_optimizer = Adahessian(self.model.parameters())
        self.num_checking = 0

        # For tracking already selected samples
        self.selection_history = {}  # Maps epoch -> selected indices
        self.selection_quality = {}  # Maps indices -> quality score
        
        # Configure subset selection parameters with optimized defaults
        # Dynamically set max_subset_size based on dataset size instead of hardcoded value
        default_max_subset = int(len(self.train_dataset) * self.args.train_frac)  # 10% of dataset
        self.max_subset_size = getattr(self.args, 'max_subset_size', default_max_subset)
        
        # Update subset less frequently for training stability and efficiency
        self.subset_refresh_frequency = getattr(self.args, 'subset_refresh_frequency', 10)
        
        # Allow configuring the ratio between diversity and coverage
        self.dpp_weight = getattr(self.args, 'dpp_weight', 0.5)
        
        # Ensure drop_detrimental is False by default
        self.args.drop_detrimental = getattr(self.args, 'drop_detrimental', False)
        
        # Batch size for inference
        self.inference_batch_size = getattr(self.args, 'inference_batch_size', self.args.batch_size * 2)
        
        # Configure parallel processing
        self.num_workers = min(getattr(self.args, 'selection_workers', multiprocessing.cpu_count() // 2), 
                               multiprocessing.cpu_count() - 1)
        
        # Mixed precision training
        self.use_mixed_precision = getattr(self.args, 'use_mixed_precision', True)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.use_mixed_precision = False
        
        # Cache for selection process
        self.cached_outputs = {}  # Maps (epoch_range, index) -> output
        self.cached_gradients = None
        self.cached_epoch = -1
        self.similarity_cache = {}  # Initialize similarity matrix cache
                
        self.args.logger.info(f"High-Performance CREST trainer initialized:")
        self.args.logger.info(f"  - Max subset size: {self.max_subset_size} samples")
        self.args.logger.info(f"  - Subset refresh frequency: Every {self.subset_refresh_frequency} epochs")
        self.args.logger.info(f"  - DPP weight (diversity vs coverage): {self.dpp_weight}")
        self.args.logger.info(f"  - Detrimental sample dropping: {self.args.drop_detrimental}")
        self.args.logger.info(f"  - Selection workers: {self.num_workers}")
        self.args.logger.info(f"  - Mixed precision training: {self.use_mixed_precision}")
        self.args.logger.info(f"  - Inference batch size: {self.inference_batch_size}")

    def _train_epoch(self, epoch: int):
        """
        Train the model for one epoch with optimized subset selection
        :param epoch: current epoch
        """
        self.model.train()
        self._reset_metrics()

        lr = self.lr_scheduler.get_last_lr()[0]
        self.args.logger.info(f"Epoch {epoch} LR {lr:.6f}")

        # Initialize subset for first epoch if not done yet
        if not hasattr(self, 'subset') or not hasattr(self, 'subset_weights'):
            subset_size = int(len(self.train_dataset) * self.args.train_frac)
            subset_size = min(subset_size, self.max_subset_size)
            self.subset = np.random.choice(self.train_indices, size=subset_size, replace=False)
            self.subset_weights = np.ones(len(self.subset)) / len(self.subset)
            self.args.logger.info(f"Initialized random subset with {len(self.subset)} samples for first epoch")

        # Only select subset at initial epoch or at fixed refresh intervals
        need_new_subset = (epoch == self.args.warm_start_epochs) or \
                          (epoch >= self.args.warm_start_epochs and 
                           (epoch - self.args.warm_start_epochs) % self.subset_refresh_frequency == 0)
        
        if need_new_subset:
            self.args.logger.info(f"Epoch {epoch}: Selecting new subset")
            selection_start_time = time.time()
            self._select_subset(epoch)
            selection_end_time = time.time()
            
            selection_time = selection_end_time - selection_start_time
            self.selection_time.update(selection_time)
            self.args.logger.info(f"Epoch {epoch}: Subset selection took {selection_time:.2f} seconds")
            
            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch, 
                    'subset_selection_time': selection_time,
                    'subset_size': len(self.subset),
                    'is_refresh_epoch': True
                })
                
        elif epoch < self.args.warm_start_epochs:
            # For warm-up epochs, use random subset
            self.args.logger.info(f"Epoch {epoch}: Warm-up epoch, using random subset")
            # Random selection with size based on args.train_frac
            subset_size = int(len(self.train_dataset) * self.args.train_frac)
            subset_size = min(subset_size, self.max_subset_size)
            self.subset = np.random.choice(self.train_indices, size=subset_size, replace=False)
            self.subset_weights = np.ones(len(self.subset)) / len(self.subset)
        
        # Set up current subset cache
        if not hasattr(self, 'current_subset') or not hasattr(self, 'current_weights'):
            self.current_subset = self.subset.copy()
            self.current_weights = self.subset_weights.copy()
        else:
            # Only update current subset if we've selected a new one
            if need_new_subset or epoch < self.args.warm_start_epochs:
                self.current_subset = self.subset.copy()
                self.current_weights = self.subset_weights.copy()
            else:
                # Reuse cached subset from previous epoch
                self.subset = self.current_subset.copy()
                self.subset_weights = self.current_weights.copy()
                
        # Always update the dataloader with the final subset
        self._update_train_loader_and_weights()
        
        # Training loop
        num_batches = len(self.train_loader)
        if num_batches == 0:
            self.args.logger.warning(f"Epoch {epoch}: Train loader is empty. Skipping epoch.")
            return
            
        self.args.logger.info(f"Epoch {epoch}: Training with {len(self.train_loader.dataset)} samples across {num_batches} batches")
        
        # Standard training loop with optimizations
        data_start = time.time()
        for batch_idx, (data, target, data_idx) in enumerate(self.train_loader):
            # Load data to device
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)
            
            # Forward and backward pass with potential mixed precision
            data = data.to(self.args.device, non_blocking=True)
            target = target.to(self.args.device, non_blocking=True)
            
            loss, train_acc = self._forward_and_backward(data, target, data_idx)
            
            # Log progress periodically
            if (batch_idx + 1) % self.args.log_interval == 0 or batch_idx == num_batches - 1:
                self.args.logger.info(
                    f"Epoch {epoch} Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss {self.train_loss.avg:.4f} Acc {self.train_acc.avg:.4f} "
                    f"DataTime {self.batch_data_time.avg:.3f} "
                    f"FwdTime {self.batch_forward_time.avg:.3f} "
                    f"BwdTime {self.batch_backward_time.avg:.3f}"
                )
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "train_loss_batch": loss.item(),
                    "train_acc_batch": train_acc,
                    "lr": lr
                })
                
            # Start timing next data loading
            data_start = time.time()
    
    def _get_train_output_efficient(self, indices=None):
        """
        Efficiently compute model outputs for given indices with batching and parallel processing
        
        :param indices: Indices to compute outputs for (if None, compute for all data)
        """
        self.model.eval()
        
        # Determine which indices to process
        if indices is None:
            indices_to_process = self.train_indices
        else:
            indices_to_process = indices
            
        # Create dataset and dataloader for efficient processing
        eval_dataset = Subset(self.train_dataset, indices_to_process)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.inference_batch_size,  # Larger batch size for inference
            shuffle=False,
            num_workers=min(self.args.num_workers * 2, 16),  # More workers for inference
            pin_memory=True
        )
        
        # Initialize arrays if they don't exist
        if not hasattr(self, 'train_output') or self.train_output.shape[0] != len(self.train_dataset):
            self.train_output = np.zeros((len(self.train_dataset), self.args.num_classes))
            self.train_softmax = np.zeros((len(self.train_dataset), self.args.num_classes))
        
        # Process batches in parallel if possible
        with torch.no_grad():
            for data, _, data_idx in eval_loader:
                data = data.to(self.args.device, non_blocking=True)
                
                # Use mixed precision for faster inference
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                else:
                    output = self.model(data)
                
                # Compute softmax efficiently on GPU before transferring to CPU
                softmax_output = F.softmax(output, dim=1)
                
                # Update arrays
                self.train_output[data_idx] = output.cpu().numpy()
                self.train_softmax[data_idx] = softmax_output.cpu().numpy()
        
        self.model.train()
        
    def _select_coreset(self, pool_indices, epoch: int):
        """
        Helper method to select a coreset within a single selection step.
        This method is optimized to run only when needed.
        """
        print(f"Starting coreset selection for pool of {len(pool_indices)} samples")
        
        # Determine target subset size
        target_subset_size = int(len(self.train_dataset) * self.args.train_frac)

        # Get predictions and targets for candidate pool
        preds = self.train_softmax[pool_indices]
        targets = self.train_target[pool_indices]
        
        # Calculate gradient features for selection
        one_hot_labels = np.zeros((len(targets), self.args.num_classes))
        one_hot_labels[np.arange(len(targets)), targets] = 1
        gradients = preds - one_hot_labels  # Gradient of cross-entropy w.r.t logits
        
        # Perform selection using mixed method
        selection_method = getattr(self.args, 'selection_method', 'mixed')
        
        # Use internal selection implementation to prevent multiple calls to subset_generator
        # This addresses the issue of multiple selection logs in the output
        if len(pool_indices) > 10000 and selection_method == "mixed":
            subset_indices, weights = self._internal_mixed_selection(
                gradients, targets, preds, target_subset_size
            )
        else:
            # Use internal selection method directly without going through the SubsetGenerator
            # to avoid unnecessary logging
            subset_indices, weights = self._internal_mixed_selection(
                gradients, 
                targets, 
                preds, 
                target_subset_size
            )
        
        # Map local indices back to global
        global_indices = pool_indices[subset_indices]
        return global_indices, weights
        
    def _internal_mixed_selection(self, features, labels, softmax_preds, subset_size):
        """
        Internal mixed selection implementation to avoid multiple calls to subset_generator
        This ensures we don't log the selection process multiple times
        """
        start_time = time.time()
        
        # Process by class for better efficiency
        classes = np.unique(labels)
        class_counts = [np.sum(labels == c) for c in classes]
        class_fractions = np.array(class_counts) / len(labels)
        
        # Calculate target size per class
        targets_per_class = np.int32(np.ceil(class_fractions * subset_size))
        
        # Ensure we don't select more than subset_size
        while np.sum(targets_per_class) > subset_size:
            idx_to_reduce = np.argmax(targets_per_class)
            targets_per_class[idx_to_reduce] -= 1
        
        class_prep_time = time.time() - start_time
        
        # Try to use the fast implementation first
        try:
            indices, weights = self._fast_mixed_selection(features, labels, softmax_preds, targets_per_class, classes)
            total_time = time.time() - start_time
            
            # Log timing information
            print(f"Fast mixed selection performance breakdown:")
            print(f"  - Total time: {total_time:.2f}s")
            print(f"  - Class preparation: {class_prep_time:.2f}s ({class_prep_time/total_time*100:.1f}%)")
            
            return indices, weights
            
        except Exception as e:
            self.args.logger.warning(f"Fast selection failed with error: {e}. Falling back to standard implementation.")
            
            # If fast implementation fails, use the original implementation as fallback
            # Selection per class (no logging)
            all_indices = []
            all_weights = []
            
            total_similarity_time = 0
            total_greedy_selection_time = 0
            total_batch_processing_time = 0
            
            for c_idx, cls in enumerate(classes):
                class_start_time = time.time()
                
                target_size = targets_per_class[c_idx]
                if target_size <= 0:
                    continue
                    
                class_mask = (labels == cls)
                class_indices = np.where(class_mask)[0]
                if len(class_indices) == 0:
                    continue
                    
                # Extract class features and normalize
                class_features = features[class_indices]
                norms = np.linalg.norm(class_features, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized_features = class_features / norms
                
                # Compute similarity matrix
                similarity_start = time.time()
                similarity_matrix = np.dot(normalized_features, normalized_features.T)
                similarity_time = time.time() - similarity_start
                total_similarity_time += similarity_time
                
                # Greedy selection
                greedy_start = time.time()
                selected = []
                remaining = list(range(len(class_indices)))
                
                # Select first element based on representation
                if remaining:
                    row_sums = np.sum(similarity_matrix, axis=1)
                    best_idx = np.argmax(row_sums)
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                
                # Select remaining elements
                for _ in range(min(target_size - 1, len(class_indices) - 1)):
                    if not remaining:
                        break
                        
                    best_idx = -1
                    best_gain = -float('inf')
                    
                    # Set batch size dynamically based on number of remaining samples
                    # instead of hardcoded 1000
                    batch_size = min(len(remaining) // 10 + 1, len(remaining))
                    
                    batch_start = time.time()
                    for i in range(0, len(remaining), batch_size):
                        batch = remaining[i:min(i + batch_size, len(remaining))]
                        
                        # Compute coverage and diversity scores
                        coverage_sims = similarity_matrix[batch, :]
                        coverage_scores = np.sum(coverage_sims, axis=1)
                        
                        if selected:
                            diversity_penalty = np.max(similarity_matrix[batch, :][:, selected], axis=1)
                        else:
                            diversity_penalty = np.zeros(len(batch))
                        
                        # Combined score with weighted diversity
                        combined_scores = (1.0 - self.dpp_weight) * coverage_scores - self.dpp_weight * diversity_penalty
                        
                        # Find best element
                        batch_best_idx = np.argmax(combined_scores)
                        batch_best_gain = combined_scores[batch_best_idx]
                        
                        if batch_best_gain > best_gain:
                            best_gain = batch_best_gain
                            best_idx = batch[batch_best_idx]
                    
                    if best_idx != -1:
                        selected.append(best_idx)
                        remaining.remove(best_idx)
                        
                    batch_time = time.time() - batch_start
                    total_batch_processing_time += batch_time
                
                greedy_selection_time = time.time() - greedy_start
                total_greedy_selection_time += greedy_selection_time
                
                # Calculate weights and map indices
                if selected:
                    row_sums = np.sum(similarity_matrix[selected, :], axis=1)
                    class_weights = row_sums / np.sum(row_sums)
                    
                    # Convert to global indices
                    global_indices = [class_indices[idx] for idx in selected]
                    
                    all_indices.extend(global_indices)
                    all_weights.extend(class_weights)
                    
                class_time = time.time() - class_start_time
                if len(class_indices) > 100:  # Only log for significant classes
                    self.args.logger.debug(f"Class {cls}: {len(class_indices)} samples -> {len(selected)} selected in {class_time:.2f}s")
                    self.args.logger.debug(f"  - Similarity matrix: {similarity_time:.2f}s")
                    self.args.logger.debug(f"  - Greedy selection: {greedy_selection_time:.2f}s")
            
            # Convert to numpy arrays
            indices = np.array(all_indices, dtype=np.int32)
            weights = np.array(all_weights, dtype=np.float32)
            
            # Normalize weights
            if len(weights) > 0:
                weights = weights / np.sum(weights)
                
            total_time = time.time() - start_time
            
            # Log timing information
            print(f"Mixed selection performance breakdown:")
            print(f"  - Total time: {total_time:.2f}s")
            print(f"  - Class preparation: {class_prep_time:.2f}s ({class_prep_time/total_time*100:.1f}%)")
            print(f"  - Similarity matrix computation: {total_similarity_time:.2f}s ({total_similarity_time/total_time*100:.1f}%)")
            print(f"  - Greedy selection: {total_greedy_selection_time:.2f}s ({total_greedy_selection_time/total_time*100:.1f}%)")
            print(f"  - Batch processing: {total_batch_processing_time:.2f}s ({total_batch_processing_time/total_time*100:.1f}%)")
                
            return indices, weights
            
        except (ImportError, AttributeError) as e:
            # Fall back to simpler implementation if submodlib is not available
            print(f"Error importing submodlib: {e}")
            raise ImportError("Submodlib library is not available for mixed selection") from e

    def _fast_mixed_selection(self, features, labels, softmax_preds, targets_per_class, classes):
        """
        Fast implementation of mixed selection using submodlib, inspired by
        facility_location_order implementation
        
        Args:
            features: Feature vectors for samples
            labels: Class labels
            softmax_preds: Model predictions
            targets_per_class: Number of samples to select per class
            classes: Unique class labels
            
        Returns:
            Tuple of (selected indices, selection weights)
        """
        try:
            from submodlib import FacilityLocationFunction, LogDeterminantFunction
            print(f"Implementing optimized selection with combined objectives")
            
            # Normalize features for similarity computation
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_features = features / norms
            
            # Track execution time
            fast_start_time = time.time()
            
            # Process each class sequentially
            all_indices = []
            all_weights = []
            
            # Process each class
            for c_idx, cls in enumerate(classes):
                target_size = targets_per_class[c_idx]
                if target_size <= 0:
                    continue
                    
                class_mask = (labels == cls)
                class_indices = np.where(class_mask)[0]
                
                if len(class_indices) == 0 or target_size == 0:
                    continue
                    
                # Extract class features
                class_features = normalized_features[class_indices]
                
                # Determine if we should use dense or sparse mode based on dataset size
                use_dense = len(class_indices) <= 10000
                
                # For mixed objective, we need to handle separately based on computation mode
                if use_dense:
                    # Compute similarity matrix for dense mode
                    similarity_matrix = np.dot(class_features, class_features.T)
                    
                    # Create facility location object for coverage
                    fl_obj = FacilityLocationFunction(
                        n=len(class_features),
                        mode="dense",
                        sijs=similarity_matrix,
                        separate_rep=False
                    )
                    
                    # Create log determinant object for diversity
                    dpp_obj = LogDeterminantFunction(
                        n=len(class_features),
                        mode="dense",
                        sijs=similarity_matrix,
                        lambdaVal=1.0
                    )
                else:
                    # For sparse mode, use data directly
                    fl_obj = FacilityLocationFunction(
                        n=len(class_features),
                        mode="sparse",
                        data=class_features,
                        metric="cosine",
                        num_neighbors=128
                    )
                    
                    dpp_obj = LogDeterminantFunction(
                        n=len(class_features),
                        mode="sparse",
                        data=class_features,
                        metric="cosine"
                    )
                
                # Use direct maximize approach for better performance
                print(f"Using efficient maximize approach with diversity weight {self.dpp_weight}")
                
                # If using pure facility location (no diversity)
                if self.dpp_weight <= 0:
                    # Use the built-in maximize for FacilityLocationFunction which is highly optimized
                    start_time = time.time()
                    greedyList = fl_obj.maximize(
                        budget=target_size,
                        optimizer="LazyGreedy",
                        stopIfZeroGain=False,
                        stopIfNegativeGain=False,
                        verbose=False
                    )
                    selection_time = time.time() - start_time
                    print(f"Pure FL maximize completed in {selection_time:.3f}s")
                    
                    # Extract selected indices and gains
                    selected = [x[0] for x in greedyList]
                    
                else:
                    # Use maximizer directly for both FL and DPP objectives
                    start_time = time.time()
                    
                    # Get FL order using efficient C++ implementation
                    fl_greedyList = fl_obj.maximize(
                        budget=target_size,
                        optimizer="LazyGreedy",
                        stopIfZeroGain=False,
                        stopIfNegativeGain=False,
                        verbose=False
                    )
                    fl_order = [x[0] for x in fl_greedyList]
                    fl_gains = [x[1] for x in fl_greedyList]
                    
                    # Get DPP order using efficient C++ implementation
                    dpp_greedyList = dpp_obj.maximize(
                        budget=target_size,
                        optimizer="LazyGreedy",
                        stopIfZeroGain=False,
                        stopIfNegativeGain=False,
                        verbose=False
                    )
                    dpp_order = [x[0] for x in dpp_greedyList]
                    dpp_gains = [x[1] for x in dpp_greedyList]
                    
                    # Combine the two orders with weighted gains
                    selected = []
                    curr_set = set()
                    fl_idx = 0
                    dpp_idx = 0
                    
                    # Interleave FL and DPP selections based on weighted gains
                    while len(selected) < target_size and (fl_idx < len(fl_order) or dpp_idx < len(dpp_order)):
                        # Skip elements already selected
                        while fl_idx < len(fl_order) and fl_order[fl_idx] in curr_set:
                            fl_idx += 1
                        while dpp_idx < len(dpp_order) and dpp_order[dpp_idx] in curr_set:
                            dpp_idx += 1
                            
                        # Check if we've exhausted either list
                        if fl_idx >= len(fl_order):
                            if dpp_idx < len(dpp_order) and dpp_order[dpp_idx] not in curr_set:
                                next_idx = dpp_order[dpp_idx]
                                dpp_idx += 1
                            else:
                                break
                        elif dpp_idx >= len(dpp_order):
                            if fl_idx < len(fl_order) and fl_order[fl_idx] not in curr_set:
                                next_idx = fl_order[fl_idx]
                                fl_idx += 1
                            else:
                                break
                        else:
                            # Both lists have candidates - compare weighted gains
                            fl_weighted = (1 - self.dpp_weight) * fl_gains[fl_idx]
                            dpp_weighted = self.dpp_weight * dpp_gains[dpp_idx]
                            
                            if fl_weighted > dpp_weighted:
                                next_idx = fl_order[fl_idx]
                                fl_idx += 1
                            else:
                                next_idx = dpp_order[dpp_idx]
                                dpp_idx += 1
                        
                        # Add the selected element
                        if next_idx not in curr_set:
                            selected.append(next_idx)
                            curr_set.add(next_idx)
                    
                    selection_time = time.time() - start_time
                    print(f"Joint maximize completed in {selection_time:.3f}s using efficient maximizer")
                
                # Convert selected list to numpy array for efficient indexing
                selected = np.array(list(selected))
                
                # Compute similarity matrix for computing cluster sizes
                if len(selected) > 0:
                    # Get the precomputed similarity matrix if we're in dense mode
                    if use_dense:
                        sim_to_selected = similarity_matrix[:, selected]
                    else:
                        # Compute similarity matrix only for selected points
                        sel_features = class_features[selected]
                        sim_to_selected = fl_obj.sijs[:, selected]  # Use precomputed similarities from FL object
                    
                    # For each point, find most similar selected point
                    closest_selected = np.argmax(sim_to_selected, axis=1)
                    
                    # Count points belonging to each cluster
                    cluster_sizes = np.zeros(len(selected))
                    class_weights = None  # Can add sample weighting here if needed
                    
                    for i in range(len(class_features)):
                        if np.max(sim_to_selected[i]) > 0:  # Only count if similarity > 0
                            cluster_sizes[closest_selected[i]] += 1 if class_weights is None else class_weights[i]
                    
                    # Ensure no zero weights
                    cluster_sizes[cluster_sizes == 0] = 1
                    
                    # Map local indices back to global indices
                    global_indices = class_indices[selected]
                    
                    if len(global_indices) > 0:
                        all_indices.extend(global_indices)
                        all_weights.extend(cluster_sizes)
            
            # Convert to numpy arrays with proper types
            indices = np.array(all_indices, dtype=np.int32)
            weights = np.array(all_weights, dtype=np.float32)
            
            # Skip normalization for better stability
            # if len(weights) > 0:
            #     weights = weights / np.sum(weights)
            
            fast_selection_time = time.time() - fast_start_time
            print(f"Fast mixed selection completed in {fast_selection_time:.2f}s, selected {len(indices)} samples")
                
            return indices, weights
            
        except (ImportError, AttributeError) as e:
            # Fall back to simpler implementation if submodlib is not available
            print(f"Error importing submodlib: {e}")
            raise ImportError("Submodlib library is not available for mixed selection") from e

    def _select_subset_impl(self, epoch: int):
        """
        Select a high-quality subset of training data, optimized for efficiency
        
        :param epoch: Current epoch
        """
        selection_start = time.time()
        self.args.logger.info(f"Epoch {epoch}: Starting optimized subset selection")
        
        # Determine target subset size
        target_subset_size = int(len(self.train_dataset) * self.args.train_frac)
        target_subset_size = min(target_subset_size, self.max_subset_size)
        
        # Use cached outputs if available from recent epochs to save computation
        use_cached = (epoch - self.cached_epoch <= 2) and hasattr(self, 'cached_pool_indices')
        
        if not use_cached:
            # Determine if we should sample based on dataset size
            # For large datasets, use sampling to speed up computation
            dataset_size = len(self.train_dataset)
            large_dataset_threshold = dataset_size  # Consider 25% of dataset size as threshold
            
            if dataset_size > large_dataset_threshold:
                # Sample size proportional to dataset size but capped
                sample_size = min(dataset_size, 50000)
                sample_indices = self._select_stratified_random(self.train_indices, sample_size)
                self._get_train_output_efficient(indices=sample_indices)
                pool_indices = sample_indices
            else:
                # For smaller datasets, compute on the entire dataset
                self._get_train_output_efficient()
                pool_indices = self.train_indices
                
            # Cache the results
            self.cached_pool_indices = pool_indices
            self.cached_epoch = epoch
        else:
            # Use cached data
            self.args.logger.info(f"Using cached model outputs from epoch {self.cached_epoch}")
            pool_indices = self.cached_pool_indices
        
        try:
            # Use optimized selection
            self.subset, self.subset_weights = self._select_coreset(pool_indices, epoch)
            
        except Exception as e:
            self.args.logger.error(f"Error during subset selection: {e}")
            self.args.logger.warning("Falling back to random subset selection")
            
            # Fallback to simple random selection
            self.subset = np.random.choice(
                self.train_indices, 
                size=min(target_subset_size, len(self.train_indices)),
                replace=False
            )
            # Option 1: Normalized weights (current approach)
            # Weights are normalized to sum to 1, which means each sample's importance
            # is relative to others and the total influence is constrained
            # self.subset_weights = np.ones(len(self.subset)) / len(self.subset)
            
            # Option 2 (alternative): Use uniform weights without normalization
            # This would give equal importance to all selected samples
            # but could affect the overall learning rate / gradient magnitude
            self.subset_weights = np.ones(len(self.subset))
        
        # Store selection history
        self.selection_history[epoch] = self.subset.copy()
        
        # Calculate quality metrics for future reference (using vectorized operations)
        if len(self.subset) > 0:
            # Use model predictions to estimate sample quality
            subset_preds = self.train_softmax[self.subset]
            subset_targets = self.train_target[self.subset]
            
            # Calculate per-sample loss (vectorized)
            sample_losses = -np.log(np.maximum(
                subset_preds[np.arange(len(subset_targets)), subset_targets], 
                1e-8
            ))
            
            # Update quality metrics (vectorized)
            quality_scores = 1.0 / (1.0 + sample_losses)
            
            # Update quality dictionary in one operation
            self.selection_quality.update(dict(zip(self.subset, quality_scores)))
        
        # Log selection stats
        total_selection_time = time.time() - selection_start
        self.args.logger.info(
            f"Epoch {epoch}: Selected {len(self.subset)} samples ({len(self.subset)/len(self.train_dataset)*100:.1f}% of dataset). "
            f"Total selection time: {total_selection_time:.2f}s"
        )
        
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'total_selection_time': total_selection_time,
                'final_subset_size': len(self.subset),
                'dpp_weight': self.dpp_weight,
                'subset_fraction': len(self.subset)/len(self.train_dataset)
            })
            
    # Override the parent class method to prevent call from SubsetTrainer
    def _select_subset(self, epoch: int, training_step: int = None):
        """
        Overridden from the parent class to ensure our implementation is used
        and to prevent multiple subset selections
        """
        
        # Configure the subset generator to be aware of the refresh frequency
        if hasattr(self.subset_generator, 'set_refresh_frequency'):
            self.subset_generator.set_refresh_frequency(self.subset_refresh_frequency)
            self.subset_generator.set_epoch(epoch)
            
        # Force a refresh since we know we need one
        if hasattr(self.subset_generator, 'force_selection_refresh'):
            self.subset_generator.force_selection_refresh()
            
        # Increment selection counter (for compatibility with parent class)
        self.num_selection += 1
        
        if self.args.use_wandb:
            wandb.log({"epoch": epoch, "training_step": training_step, "num_selection": self.num_selection})
            
        # Handle dataset caching (from parent class)
        if self.args.cache_dataset:
            self.train_dataset.clean()
            self.train_dataset.cache()
            
        # Use our optimized subset selection
        selection_start = time.time()
        self._select_subset_impl(epoch)
        selection_time = time.time() - selection_start
        
        self.args.logger.info(f"Epoch {epoch}: Complete subset selection took {selection_time:.2f} seconds")
        
        # Log to wandb
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'selection_time': selection_time,
                'subset_size': len(self.subset),
                'is_refresh_epoch': True
            })


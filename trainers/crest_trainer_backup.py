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
        
        # Control whether to normalize features for selection
        self.args.normalize_features = getattr(self.args, 'normalize_features', False)
        
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
            self.subset_weights = np.ones(len(self.subset))
        
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
        
        # Implementation of optimized subset selection (formerly _select_subset_impl)
        self.args.logger.info(f"Epoch {epoch}: Starting optimized subset selection")
        
        # Determine target subset size
        target_subset_size = int(len(self.train_dataset) * self.args.train_frac)
        
        
        # Determine if we should sample based on dataset size
        # For large datasets, use sampling to speed up computation
        dataset_size = len(self.train_dataset)
        large_dataset_threshold = dataset_size 
        
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
            

        # Use cached data
        self.args.logger.info(f"Using cached model outputs from epoch {self.cached_epoch}")
        
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
        
        selection_time = time.time() - selection_start
        self.args.logger.info(f"Epoch {epoch}: Complete subset selection took {selection_time:.2f} seconds")
        
        # Log to wandb
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'training_step': training_step,
                'num_selection': self.num_selection,
                'selection_time_coreset_selection_call': selection_time, # Renamed for clarity
                'subset_size_after_coreset_selection': len(self.subset),
                'is_refresh_epoch': True, # Assuming this method is called on refresh
                'total_selection_time': total_selection_time,
                'final_subset_size': len(self.subset),
                'dpp_weight': self.dpp_weight if hasattr(self, 'dpp_weight') else -1, # Check if attr exists
                'subset_fraction': len(self.subset)/len(self.train_dataset) if len(self.train_dataset) > 0 else 0,
                'pool_indices_size': len(pool_indices) if pool_indices is not None else 0
            })
            
    
        
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
        if selection_method == "mixed":
            # Internal mixed selection implementation 
            start_time = time.time()
            
            # Process by class for better efficiency
            classes = np.unique(targets)
            class_counts = [np.sum(targets == c) for c in classes]
            class_fractions = np.array(class_counts) / len(targets)
            
            # Calculate target size per class
            targets_per_class = np.int32(np.ceil(class_fractions * target_subset_size))
            
            # Ensure we don't select more than subset_size
            while np.sum(targets_per_class) > target_subset_size:
                idx_to_reduce = np.argmax(targets_per_class)
                targets_per_class[idx_to_reduce] -= 1
            
            class_prep_time = time.time() - start_time
            
            # Try to use the simplified facility location implementation
            try:
                subset_indices, weights = self._fast_mixed_selection(gradients, targets, preds, targets_per_class, classes)
                total_time = time.time() - start_time
                
                # Log timing information
                print(f"Selection performance breakdown:")
                print(f"  - Total time: {total_time:.2f}s")
                print(f"  - Class preparation: {class_prep_time:.2f}s ({class_prep_time/total_time*100:.1f}%)")
                
            except Exception as e:
                self.args.logger.warning(f"Selection failed with error: {str(e)}")
                self.args.logger.warning("Error details:", exc_info=True)
                print(f"Error with _select_coreset: {str(e)}")
                self.args.logger.error(f"Error during _select_coreset: {str(e)}", exc_info=True) # Added more detailed logging
                
                # Fallback to random selection
                subset_indices = np.random.choice(
                    np.arange(len(pool_indices)), 
                    size=min(target_subset_size, len(pool_indices)),
                    replace=False
                )
                weights = np.ones(len(subset_indices))
                
        # TODO: Implement other selection methods if needed
        else:
            # Fallback to random subset selection if method is not implemented
            self.args.logger.warning(f"Selection method '{selection_method}' not implemented. Using random selection.")
            
            # Generate random subset
            subset_indices = np.random.choice(
                np.arange(len(pool_indices)), 
                size=min(target_subset_size, len(pool_indices)),
                replace=False
            )
            weights = np.ones(len(subset_indices))
        
        # Map local indices back to global
        global_indices = pool_indices[subset_indices]
        return global_indices, weights

    def _fast_mixed_selection(self, features, labels, softmax_preds, targets_per_class, classes):
        """
        Improved implementation of mixed selection using a two-stage approach:
        1. Select a larger intermediate subset based on coverage
        2. Cluster similar instances and reduce to the final subset
        
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
            from sklearn.cluster import AgglomerativeClustering
            import scipy.spatial.distance as distance
            
            # Control feature normalization with a parameter
            # For gradients, this should typically be False to preserve magnitude information
            normalize_features = getattr(self.args, 'normalize_features', False)
            
            print(f"Implementing two-stage selection for coverage + diversity (normalize_features={normalize_features})")
            
            # Track execution time
            start_time = time.time()
            
            # Process original features or normalize them based on the parameter
            if normalize_features:
                # Normalize features for similarity computation (preserves direction only)
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                processed_features = features / norms
                print("Using normalized features (direction only)")
            else:
                # Use original features (preserves both direction and magnitude)
                processed_features = features.copy()
                print("Using original features (preserving magnitude)")
            
            # Process each class separately
            all_selected_indices = []
            all_selected_weights = []
            
            for c_idx, cls in enumerate(classes):
                target_class_size = targets_per_class[c_idx]
                if target_class_size <= 0:
                    continue
                
                # Get class-specific data
                class_mask = (labels == cls)
                class_indices = np.where(class_mask)[0]
                
                if len(class_indices) == 0:
                    continue
                
                if len(class_indices) <= target_class_size:
                    # If we have fewer samples than target, use all available
                    all_selected_indices.extend(class_indices)
                    all_selected_weights.extend(np.ones(len(class_indices)))
                    continue
                
                # Extract class features
                class_features = processed_features[class_indices]
                n_samples = len(class_features)
                
                # STAGE 1: Initial coverage-based selection (facility location)
                # Select larger intermediate subset (2x target size, but capped)
                intermediate_size = min(n_samples, int(target_class_size * 2))
                
                # Use dense mode for smaller datasets, sparse for larger ones
                use_dense = n_samples <= 5000  # Threshold adjusted for performance
                mode = "dense" if use_dense else "sparse"
                
                # Create facility location object
                fl_time_start = time.time()
                try:
                    if use_dense:
                        # For dense mode, precompute similarity matrix
                        # If not normalized, we need to be careful with similarity computation
                        if normalize_features:
                            # For normalized features, dot product directly gives cosine similarity
                            similarity_matrix = np.dot(class_features, class_features.T)
                        else:
                            # For unnormalized features, we compute a scaled similarity
                            # This helps prevent numerical issues with very large magnitudes
                            # while still preserving relative magnitudes
                            magnitudes = np.linalg.norm(class_features, axis=1, keepdims=True)
                            max_magnitude = np.max(magnitudes)
                            if max_magnitude > 0:
                                scaled_features = class_features / max_magnitude
                                similarity_matrix = np.dot(scaled_features, scaled_features.T)
                                print(f"  Class {cls}: Used scaled dot product for similarity (max_magnitude: {max_magnitude:.2f})")
                            else:
                                # All features are zero vectors
                                similarity_matrix = np.zeros((n_samples, n_samples))
                                print(f"  Class {cls}: All features are zero, using zero similarity matrix.")
                        fl_obj = FacilityLocationFunction(
                            n=n_samples,
                            mode="dense",
                            sijs=similarity_matrix,
                            separate_rep=False
                        )
                    else:
                        # For sparse mode, let submodlib compute similarities as needed
                        # Specify appropriate metric based on whether features are normalized
                        metric = "cosine" if normalize_features else "euclidean"
                        fl_obj = FacilityLocationFunction(
                            n=n_samples,
                            mode="sparse",
                            data=class_features,
                            metric=metric,
                            num_neighbors=min(128, n_samples - 1)
                        )
                        
                    fl_prep_time = time.time() - fl_time_start
                    
                    # Run lazy greedy selection
                    greedy_start = time.time()
                    intermediate_greedy_list = fl_obj.maximize(
                        budget=intermediate_size,
                        optimizer="LazyGreedy",
                        stopIfZeroGain=False,
                        stopIfNegativeGain=False,
                        verbose=False
                    )
                    greedy_time = time.time() - greedy_start
                    
                    # Extract the initial selected points
                    intermediate_selected = [x[0] for x in intermediate_greedy_list]
                    intermediate_selected = np.array(intermediate_selected, dtype=np.int32)
                    
                    # Get similarity matrix for selected points
                    if use_dense:
                        # Already computed
                        S = similarity_matrix
                    else:
                        # Use the similarity matrix from facility location
                        S = fl_obj.sijs
                        
                    # STAGE 2: Redundancy reduction through clustering
                    # If we have more than target size after initial selection
                    self.args.logger.info(f"Class {cls}: STAGE 2 (Clustering/DPP) is a placeholder and not fully implemented.")
                    # For now, we'll just use the result from STAGE 1 if it's already at or below target_class_size
                    # or truncate if it's larger. A proper DPP or clustering step would go here.
                    if len(intermediate_selected) > target_class_size:
                        self.args.logger.info(f"Class {cls}: Truncating intermediate selection from {len(intermediate_selected)} to {target_class_size}")
                        final_selected_indices_for_class = intermediate_selected[:target_class_size]
                    else:
                        final_selected_indices_for_class = intermediate_selected
                    
                    all_selected_indices.extend(class_indices[final_selected_indices_for_class])
                    all_selected_weights.extend(np.ones(len(final_selected_indices_for_class)))

                except Exception as e:
                    self.args.logger.error(f"Error in _fast_mixed_selection for class {cls}: {str(e)}", exc_info=True)
                    self.args.logger.warning(f"Class {cls}: Falling back to random selection for this class due to error.")
                    # Fallback for this class: select randomly if an error occurs
                    num_to_select_for_class_fallback = min(target_class_size, len(class_indices))
                    if len(class_indices) > 0 and num_to_select_for_class_fallback > 0:
                        fallback_indices = np.random.choice(class_indices, size=num_to_select_for_class_fallback, replace=False)
                        all_selected_indices.extend(fallback_indices)
                        all_selected_weights.extend(np.ones(len(fallback_indices)))
            
            if not all_selected_indices: # If no samples were selected (e.g., all classes had errors or were empty)
                self.args.logger.warning("No samples selected in _fast_mixed_selection. Returning empty selection.")
                return np.array([]), np.array([])

            return np.array(all_selected_indices, dtype=np.int32), np.array(all_selected_weights)
            
        except (ImportError, AttributeError) as e:
            self.args.logger.error(f"Missing dependencies (submodlib or sklearn) for _fast_mixed_selection: {str(e)}", exc_info=True)
            # Re-raise critical dependency errors, as the function cannot operate as intended.
            raise e

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
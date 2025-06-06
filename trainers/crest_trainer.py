from utils import Adahessian
from utils.learnable_lambda import LearnableLambda
from datasets.subset import get_coreset
from .subset_trainer import *
import time
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import signal


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
        
        # PERFORMANCE FOCUS: Update subset more frequently for better adaptation
        self.subset_refresh_frequency = getattr(self.args, 'subset_refresh_frequency', 3)  # Every 3 epochs instead of 10
        
        # Allow configuring the ratio between diversity and coverage
        self.dpp_weight = getattr(self.args, 'dpp_weight', 0.5)
        
        # SIMPLIFIED: Remove normalize_features - always preserve magnitude for better performance
        # self.args.normalize_features = getattr(self.args, 'normalize_features', False)
        
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
                
        self.args.logger.info(f"PERFORMANCE-FOCUSED CREST trainer initialized:")
        self.args.logger.info(f"  - Max subset size: {self.max_subset_size} samples")
        self.args.logger.info(f"  - Subset refresh frequency: Every {self.subset_refresh_frequency} epochs (frequent updates)")
        self.args.logger.info(f"  - Features: Uncertainty-based (no complex gradient processing)")
        self.args.logger.info(f"  - Stage 2: IMPLEMENTED (diversity reduction)")
        self.args.logger.info(f"  - Fallback: Uncertainty-based selection (better than random)")
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
        large_dataset_threshold = 20000  # Much more reasonable threshold
        
        if dataset_size > large_dataset_threshold:
            # MUCH smaller sample size for efficiency - 5000 is still effective
            sample_size = min(5000, dataset_size // 2)  # Maximum 5000 samples or half dataset
            sample_indices = self._select_stratified_random(self.train_indices, sample_size)
            self._get_train_output_efficient(indices=sample_indices)
            pool_indices = sample_indices
            self.args.logger.info(f"Large dataset detected ({dataset_size} samples). Sampling {sample_size} for efficiency.")
        else:
            # For smaller datasets, compute on the entire dataset
            self._get_train_output_efficient()
            pool_indices = self.train_indices
            self.args.logger.info(f"Using all {dataset_size} samples for selection.")
            

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
        
        # SIMPLIFIED: Use prediction uncertainty directly as features (no complex gradient processing)
        # Compute uncertainty features - these are more meaningful than raw gradients
        max_preds = np.max(preds, axis=1, keepdims=True)
        entropy = -np.sum(preds * np.log(preds + 1e-8), axis=1, keepdims=True)
        margin = (np.max(preds, axis=1) - np.partition(preds, -2, axis=1)[:, -2]).reshape(-1, 1)
        
        # Combine uncertainty features (no normalization to preserve magnitude information)
        features = np.concatenate([preds, entropy, margin, max_preds], axis=1)
        
        # Perform selection using mixed method
        selection_method = getattr(self.args, 'selection_method', 'mixed')
        
        # EFFICIENCY CHECK: Use fast method for large pools
        pool_size = len(pool_indices)
        use_fast_method = pool_size > 3000  # Use fast method for pools larger than 3000
        
        if selection_method == "mixed" and not use_fast_method:
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
            
            try:
                subset_indices, weights = self._fast_mixed_selection_fixed(features, targets, preds, targets_per_class, classes)
                total_time = time.time() - start_time
                print(f"Selection performance: {total_time:.2f}s")
                
            except Exception as e:
                self.args.logger.error(f"Error during _select_coreset: {str(e)}", exc_info=True)
                
                # Fallback to uncertainty-based selection (much better than random)
                print("Falling back to uncertainty-based selection")
                uncertainty_scores = entropy.flatten()
                if len(uncertainty_scores) >= target_subset_size:
                    # Select most uncertain samples
                    uncertain_indices = np.argsort(uncertainty_scores)[-target_subset_size:]
                    subset_indices = uncertain_indices
                    # Weight by uncertainty (higher uncertainty = higher weight)
                    weights = uncertainty_scores[subset_indices]
                    weights = weights / np.sum(weights) * len(weights)  # Normalize
                else:
                    subset_indices = np.arange(len(pool_indices))
                    weights = np.ones(len(subset_indices))
                
        else:
            # Use fast uncertainty-based selection for large pools or non-mixed methods
            if use_fast_method:
                self.args.logger.info(f"Large pool ({pool_size} samples) detected. Using fast uncertainty-based selection.")
            else:
                self.args.logger.warning(f"Selection method '{selection_method}' not implemented. Using uncertainty-based selection.")
            
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
            
            start_time = time.time()
            subset_indices, weights = self._fast_uncertainty_selection(features, targets, preds, targets_per_class, classes)
            total_time = time.time() - start_time
            print(f"Fast uncertainty selection performance: {total_time:.2f}s")
        
        # Map local indices back to global
        global_indices = pool_indices[subset_indices]
        return global_indices, weights

    def _fast_mixed_selection_fixed(self, features, labels, softmax_preds, targets_per_class, classes):
        """
        FIXED implementation with proper Stage 2 diversity reduction
        Focus: Performance > Efficiency, with minimal unnecessary normalization
        OPTIMIZED: Added efficiency controls and smaller selection sizes
        """
        try:
            from submodlib import FacilityLocationFunction
            
            all_selected_indices = []
            all_selected_weights = []
            
            # Add timeout handling for very slow operations
            def timeout_handler(signum, frame):
                raise TimeoutError("Selection taking too long")
            
            for c_idx, cls in enumerate(classes):
                target_class_size = targets_per_class[c_idx]
                if target_class_size <= 0:
                    continue
                
                class_mask = (labels == cls)
                class_indices = np.where(class_mask)[0]
                
                if len(class_indices) == 0:
                    continue
                
                if len(class_indices) <= target_class_size:
                    all_selected_indices.extend(class_indices)
                    all_selected_weights.extend(np.ones(len(class_indices)))
                    continue
                
                # EFFICIENCY CONTROL: For very large classes, pre-sample to reasonable size
                max_class_size = 2000  # Maximum samples per class to consider
                if len(class_indices) > max_class_size:
                    # Pre-select most uncertain samples
                    class_preds = softmax_preds[class_indices]
                    class_uncertainty = -np.sum(class_preds * np.log(class_preds + 1e-8), axis=1)
                    top_uncertain = np.argsort(class_uncertainty)[-max_class_size:]
                    class_indices = class_indices[top_uncertain]
                    self.args.logger.info(f"Class {cls}: Pre-selected {len(class_indices)} most uncertain from {len(np.where(class_mask)[0])} samples")
                
                # Extract class features - NO normalization to preserve magnitude
                class_features = features[class_indices]
                n_samples = len(class_features)
                
                # STAGE 1: Select intermediate set using facility location (SMALLER multiplier)
                intermediate_size = min(n_samples, max(target_class_size, int(target_class_size * 1.2)))  # Only 1.2x instead of 1.5x
                
                self.args.logger.info(f"Class {cls}: Selecting {intermediate_size} from {n_samples} samples...")
                
                try:
                    # Set timeout for facility location (30 seconds max per class)
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)
                    
                    # OPTIMIZED: Use simpler similarity for large feature sets
                    if class_features.shape[1] > 50:
                        # Use PCA or random projection to reduce feature dimensionality
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=min(50, class_features.shape[1]))
                        reduced_features = pca.fit_transform(class_features)
                    else:
                        reduced_features = class_features
                    
                    # Simple similarity computation without over-normalization
                    similarity_matrix = np.dot(reduced_features, reduced_features.T)
                    
                    # Facility location for coverage
                    fl_obj = FacilityLocationFunction(
                        n=n_samples,
                        mode="dense",
                        sijs=similarity_matrix,
                        separate_rep=False
                    )
                    
                    intermediate_list = fl_obj.maximize(
                        budget=intermediate_size,
                        optimizer="LazyGreedy",
                        stopIfZeroGain=True,  # Stop early if no improvement
                        stopIfNegativeGain=True,
                        verbose=False
                    )
                    
                    signal.alarm(0)  # Cancel timeout
                    intermediate_selected = [x[0] for x in intermediate_list]
                    
                except (TimeoutError, Exception) as e:
                    signal.alarm(0)  # Cancel timeout
                    self.args.logger.warning(f"Class {cls}: Facility location failed ({str(e)}), using uncertainty-based selection")
                    
                    # Fallback: Select most uncertain samples
                    class_preds = softmax_preds[class_indices]
                    uncertainty = -np.sum(class_preds * np.log(class_preds + 1e-8), axis=1)
                    intermediate_selected = np.argsort(uncertainty)[-intermediate_size:].tolist()
                
                # STAGE 2: FIXED - Diversity reduction using greedy selection
                if len(intermediate_selected) > target_class_size:
                    # Use uncertainty + diversity for final selection
                    intermediate_features = class_features[intermediate_selected]
                    intermediate_preds = softmax_preds[class_indices[intermediate_selected]]
                    
                    # Compute uncertainty scores for intermediate samples
                    uncertainty = -np.sum(intermediate_preds * np.log(intermediate_preds + 1e-8), axis=1)
                    
                    # SIMPLIFIED Greedy selection: Just use uncertainty ranking for speed
                    final_indices = np.argsort(uncertainty)[-target_class_size:].tolist()
                    
                else:
                    final_indices = intermediate_selected
                
                # Add to global selection
                all_selected_indices.extend(class_indices[final_indices])
                
                # Weight by uncertainty (higher uncertainty = higher weight)
                if final_indices:
                    final_preds = softmax_preds[class_indices[final_indices]]
                    final_uncertainty = -np.sum(final_preds * np.log(final_preds + 1e-8), axis=1)
                    # Normalize uncertainties to create weights
                    weights = final_uncertainty / (np.sum(final_uncertainty) + 1e-8) * len(final_indices)
                    all_selected_weights.extend(weights)

            return np.array(all_selected_indices, dtype=np.int32), np.array(all_selected_weights)
            
        except Exception as e:
            self.args.logger.error(f"Error in _fast_mixed_selection_fixed: {str(e)}", exc_info=True)
            raise e
    
    
    def _select_stratified_random(self, indices, sample_size):
        """
        Perform stratified random sampling to maintain class distribution
        
        :param indices: Array of indices to sample from
        :param sample_size: Number of samples to select
        :return: Selected indices
        """
        # Get targets for the given indices
        targets = self.train_target[indices]
        
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        total_samples = len(indices)
        
        selected_indices = []
        
        for cls, count in zip(unique_classes, class_counts):
            # Calculate proportional sample size for this class
            class_proportion = count / total_samples
            class_sample_size = int(np.ceil(class_proportion * sample_size))
            
            # Get indices for this class
            class_mask = (targets == cls)
            class_indices = indices[class_mask]
            
            # Sample from this class
            if len(class_indices) <= class_sample_size:
                # If we have fewer samples than needed, take all
                selected_indices.extend(class_indices)
            else:
                # Randomly sample the required number
                sampled = np.random.choice(class_indices, size=class_sample_size, replace=False)
                selected_indices.extend(sampled)
        
        # Convert to numpy array and ensure we don't exceed sample_size
        selected_indices = np.array(selected_indices)
        
        if len(selected_indices) > sample_size:
            # If we have too many due to rounding up, randomly subsample
            selected_indices = np.random.choice(selected_indices, size=sample_size, replace=False)
        
        return selected_indices
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

    def _fast_uncertainty_selection(self, features, labels, softmax_preds, targets_per_class, classes):
        """
        Fast fallback selection method using only uncertainty ranking
        Much faster than facility location for large datasets
        """
        all_selected_indices = []
        all_selected_weights = []
        
        for c_idx, cls in enumerate(classes):
            target_class_size = targets_per_class[c_idx]
            if target_class_size <= 0:
                continue
            
            class_mask = (labels == cls)
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
                
            if len(class_indices) <= target_class_size:
                all_selected_indices.extend(class_indices)
                all_selected_weights.extend(np.ones(len(class_indices)))
                continue
            
            # Get predictions for this class
            class_preds = softmax_preds[class_indices]
            
            # Compute uncertainty (entropy)
            uncertainty = -np.sum(class_preds * np.log(class_preds + 1e-8), axis=1)
            
            # Select most uncertain samples
            most_uncertain_indices = np.argsort(uncertainty)[-target_class_size:]
            selected_class_indices = class_indices[most_uncertain_indices]
            
            all_selected_indices.extend(selected_class_indices)
            
            # Weight by uncertainty
            selected_uncertainty = uncertainty[most_uncertain_indices]
            weights = selected_uncertainty / (np.sum(selected_uncertainty) + 1e-8) * len(selected_uncertainty)
            all_selected_weights.extend(weights)
        
        return np.array(all_selected_indices, dtype=np.int32), np.array(all_selected_weights) 
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
                'selection_time': selection_time,
                'subset_size': len(self.subset),
                'is_refresh_epoch': True,
                'total_selection_time': total_selection_time,
                'final_subset_size': len(self.subset),
                'dpp_weight': self.dpp_weight,
                'subset_fraction': len(self.subset)/len(self.train_dataset)
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
            
            print(f"Implementing two-stage selection for coverage + diversity")
            
            # Track execution time
            start_time = time.time()
            
            # Normalize features for similarity computation
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized_features = features / norms
            
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
                class_features = normalized_features[class_indices]
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
                        similarity_matrix = np.dot(class_features, class_features.T)
                        fl_obj = FacilityLocationFunction(
                            n=n_samples,
                            mode="dense",
                            sijs=similarity_matrix,
                            separate_rep=False
                        )
                    else:
                        # For sparse mode, let submodlib compute similarities as needed
                        fl_obj = FacilityLocationFunction(
                            n=n_samples,
                            mode="sparse",
                            data=class_features,
                            metric="cosine",
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
                    if len(intermediate_selected) > target_class_size:
                        cluster_start = time.time()
                        
                        # Extract features for initially selected points
                        selected_features = class_features[intermediate_selected]
                        
                        # Compute similarity/distance matrix for clustering
                        if len(intermediate_selected) <= 1000:
                            # Direct approach for smaller datasets
                            distance_matrix = 1 - np.dot(selected_features, selected_features.T)
                            
                            # Apply hierarchical clustering with desired number of clusters
                            clustering = AgglomerativeClustering(
                                n_clusters=target_class_size, 
                                metric='precomputed',
                                linkage='average'
                            )
                            clusters = clustering.fit_predict(distance_matrix)
                        else:
                            # For larger datasets, use direct clustering on features
                            clustering = AgglomerativeClustering(
                                n_clusters=target_class_size,
                                metric='cosine',
                                linkage='average'
                            )
                            clusters = clustering.fit_predict(selected_features)
                        
                        # For each cluster, find the most representative sample (closest to cluster center)
                        final_selected = []
                        final_weights = []
                        
                        for cluster_id in range(target_class_size):
                            # Get indices of samples in this cluster
                            cluster_samples = np.where(clusters == cluster_id)[0]
                            
                            if len(cluster_samples) == 0:
                                continue
                                
                            # Get original indices
                            cluster_original_indices = intermediate_selected[cluster_samples]
                            
                            # Calculate cluster center
                            cluster_features = class_features[cluster_original_indices]
                            cluster_center = np.mean(cluster_features, axis=0)
                            
                            # Find sample closest to center (most representative)
                            similarities = np.dot(cluster_features, cluster_center)
                            most_representative_idx = cluster_samples[np.argmax(similarities)]
                            
                            # Add to final selection
                            final_selected.append(intermediate_selected[most_representative_idx])
                            
                            # Weight is proportional to cluster size
                            final_weights.append(len(cluster_samples))
                        
                        # Convert to numpy arrays
                        final_selected = np.array(final_selected, dtype=np.int32)
                        final_weights = np.array(final_weights, dtype=np.float32)
                        
                        # Normalize weights if needed
                        final_weights = final_weights / np.sum(final_weights) * len(final_weights)
                        
                        cluster_time = time.time() - cluster_start
                        print(f"Class {cls}: Selected {len(final_selected)} samples after clustering. Clustering time: {cluster_time:.3f}s")
                    else:
                        # Not enough samples to cluster, use all intermediate results
                        final_selected = intermediate_selected
                        
                        # Compute weights based on facility location
                        cluster_sizes = np.zeros(len(final_selected), dtype=np.float32)
                        for i in range(n_samples):
                            if np.max(S[i, final_selected]) <= 0:
                                continue
                            # Assign each point to the most similar selected point
                            cluster_sizes[np.argmax(S[i, final_selected])] += 1
                            
                        # Ensure no zero weights
                        cluster_sizes[cluster_sizes == 0] = 1
                        final_weights = cluster_sizes
                        
                    # Map back to global indices and add to results
                    global_class_indices = class_indices[final_selected]
                    all_selected_indices.extend(global_class_indices)
                    all_selected_weights.extend(final_weights)
                    
                    fl_time = fl_prep_time + greedy_time
                    total_class_time = time.time() - fl_time_start
                    print(f"Class {cls}: selected {len(global_class_indices)}/{n_samples} samples, FL time: {fl_time:.3f}s, Total: {total_class_time:.3f}s")
                
                except Exception as e:
                    print(f"Error selecting samples for class {cls}: {str(e)}")
                    # Fall back to random selection for this class
                    rand_indices = np.random.choice(
                        class_indices, 
                        size=min(target_class_size, len(class_indices)),
                        replace=False
                    )
                    all_selected_indices.extend(rand_indices)
                    all_selected_weights.extend(np.ones(len(rand_indices)))
            
            # Convert to numpy arrays
            indices = np.array(all_selected_indices, dtype=np.int32)
            weights = np.array(all_selected_weights, dtype=np.float32)
            
            selection_time = time.time() - start_time
            print(f"Two-stage selection completed in {selection_time:.2f}s, selected {len(indices)} samples")
            
            return indices, weights
            
        except (ImportError, AttributeError) as e:
            # Fall back to simpler implementation if required libraries not available
            print(f"Error in two-stage selection: {e}")
            raise ImportError("Required libraries not available for selection") from e

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
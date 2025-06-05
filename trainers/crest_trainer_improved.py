from utils import Adahessian
from utils.learnable_lambda import LearnableLambda
from datasets.subset import get_coreset
from .subset_trainer import *
import time
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import numpy as np


class ImprovedCRESTTrainer(SubsetTrainer):
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
            self.args.log_interval = 10

        # Performance metrics
        self.approx_time = AverageMeter()
        self.selection_time = AverageMeter()
        self.similarity_time = AverageMeter()
        
        # Gradient approximation
        self.gradient_approx_optimizer = Adahessian(self.model.parameters())
        self.num_checking = 0

        # For tracking selection history
        self.selection_history = {}
        self.selection_quality = {}
        
        # Configure subset selection parameters with improved defaults
        default_max_subset = int(len(self.train_dataset) * self.args.train_frac)
        self.max_subset_size = getattr(self.args, 'max_subset_size', default_max_subset)
        self.subset_refresh_frequency = getattr(self.args, 'subset_refresh_frequency', 5)  # More frequent updates
        
        # Selection strategy parameters
        self.uncertainty_weight = getattr(self.args, 'uncertainty_weight', 0.4)
        self.diversity_weight = getattr(self.args, 'diversity_weight', 0.3)
        self.coverage_weight = getattr(self.args, 'coverage_weight', 0.3)
        
        # Feature processing parameters
        self.use_gradnorm_features = getattr(self.args, 'use_gradnorm_features', True)
        self.feature_combination = getattr(self.args, 'feature_combination', 'concat')  # 'concat', 'weighted'
        
        # Quality scoring parameters
        self.quality_metric = getattr(self.args, 'quality_metric', 'combined')  # 'loss', 'uncertainty', 'combined'
        
        # Mixed precision and efficiency settings
        self.use_mixed_precision = getattr(self.args, 'use_mixed_precision', True)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.use_mixed_precision = False
        
        # Cache for selection process
        self.cached_outputs = {}
        self.cached_gradients = None
        self.cached_epoch = -1
        
        self.args.logger.info(f"Improved CREST trainer initialized:")
        self.args.logger.info(f"  - Subset refresh frequency: Every {self.subset_refresh_frequency} epochs")
        self.args.logger.info(f"  - Selection weights - Uncertainty: {self.uncertainty_weight}, Diversity: {self.diversity_weight}, Coverage: {self.coverage_weight}")
        self.args.logger.info(f"  - Feature combination: {self.feature_combination}")
        self.args.logger.info(f"  - Quality metric: {self.quality_metric}")

    def _train_epoch(self, epoch: int):
        """Train the model for one epoch with improved subset selection"""
        self.model.train()
        self._reset_metrics()

        lr = self.lr_scheduler.get_last_lr()[0]
        self.args.logger.info(f"Epoch {epoch} LR {lr:.6f}")

        # Initialize subset for first epoch if not done yet
        if not hasattr(self, 'subset') or not hasattr(self, 'subset_weights'):
            subset_size = int(len(self.train_dataset) * self.args.train_frac)
            subset_size = min(subset_size, self.max_subset_size)
            self.subset = np.random.choice(self.train_indices, size=subset_size, replace=False)
            self.subset_weights = np.ones(len(self.subset))
            self.args.logger.info(f"Initialized random subset with {len(self.subset)} samples for first epoch")

        # Select new subset based on improved criteria
        need_new_subset = (epoch == self.args.warm_start_epochs) or \
                          (epoch >= self.args.warm_start_epochs and 
                           (epoch - self.args.warm_start_epochs) % self.subset_refresh_frequency == 0)
        
        if need_new_subset:
            self.args.logger.info(f"Epoch {epoch}: Selecting new subset with improved strategy")
            selection_start_time = time.time()
            self._improved_subset_selection(epoch)
            selection_end_time = time.time()
            
            selection_time = selection_end_time - selection_start_time
            self.selection_time.update(selection_time)
            self.args.logger.info(f"Epoch {epoch}: Improved subset selection took {selection_time:.2f} seconds")
                
        elif epoch < self.args.warm_start_epochs:
            # For warm-up epochs, use stratified random subset
            self.args.logger.info(f"Epoch {epoch}: Warm-up epoch, using stratified random subset")
            subset_size = int(len(self.train_dataset) * self.args.train_frac)
            subset_size = min(subset_size, self.max_subset_size)
            self.subset, self.subset_weights = self._stratified_random_selection(subset_size)
        
        # Update dataloader with new subset
        self._update_train_loader_and_weights()
        
        # Training loop (same as parent class)
        num_batches = len(self.train_loader)
        if num_batches == 0:
            self.args.logger.warning(f"Epoch {epoch}: Train loader is empty. Skipping epoch.")
            return
            
        self.args.logger.info(f"Epoch {epoch}: Training with {len(self.train_loader.dataset)} samples across {num_batches} batches")
        
        # Standard training loop
        data_start = time.time()
        for batch_idx, (data, target, data_idx) in enumerate(self.train_loader):
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)
            
            data = data.to(self.args.device, non_blocking=True)
            target = target.to(self.args.device, non_blocking=True)
            
            loss, train_acc = self._forward_and_backward(data, target, data_idx)
            
            if (batch_idx + 1) % self.args.log_interval == 0 or batch_idx == num_batches - 1:
                self.args.logger.info(
                    f"Epoch {epoch} Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss {self.train_loss.avg:.4f} Acc {self.train_acc.avg:.4f} "
                    f"DataTime {self.batch_data_time.avg:.3f} "
                    f"FwdTime {self.batch_forward_time.avg:.3f} "
                    f"BwdTime {self.batch_backward_time.avg:.3f}"
                )
            
            data_start = time.time()

    def _improved_subset_selection(self, epoch: int):
        """Improved subset selection combining multiple criteria"""
        self.args.logger.info(f"Epoch {epoch}: Starting improved subset selection")
        
        # Get model outputs
        self._get_train_output_efficient()
        
        # Calculate target subset size
        target_subset_size = int(len(self.train_dataset) * self.args.train_frac)
        
        try:
            # Compute improved features for selection
            features = self._compute_improved_features()
            
            # Compute sample quality scores
            quality_scores = self._compute_sample_quality()
            
            # Select subset using combined strategy
            self.subset, self.subset_weights = self._combined_selection_strategy(
                features, quality_scores, target_subset_size, epoch
            )
            
        except Exception as e:
            self.args.logger.error(f"Error during improved subset selection: {e}")
            self.args.logger.warning("Falling back to stratified random selection")
            
            # Fallback to stratified random selection
            self.subset, self.subset_weights = self._stratified_random_selection(target_subset_size)
        
        # Store selection history
        self.selection_history[epoch] = self.subset.copy()
        
        self.args.logger.info(
            f"Epoch {epoch}: Selected {len(self.subset)} samples ({len(self.subset)/len(self.train_dataset)*100:.1f}% of dataset)"
        )

    def _compute_improved_features(self):
        """Compute improved feature representations for selection"""
        
        if self.use_gradnorm_features:
            # Compute gradient-based features
            grad_features = self._compute_gradient_features()
        else:
            grad_features = None
            
        # Compute prediction-based features
        pred_features = self._compute_prediction_features()
        
        # Combine features
        if self.feature_combination == 'concat' and grad_features is not None:
            features = np.concatenate([pred_features, grad_features], axis=1)
        elif self.feature_combination == 'weighted' and grad_features is not None:
            # Weighted combination with learnable or fixed weights
            alpha = 0.6  # Weight for prediction features
            beta = 0.4   # Weight for gradient features
            
            # Normalize features before combining
            pred_norm = pred_features / (np.linalg.norm(pred_features, axis=1, keepdims=True) + 1e-8)
            grad_norm = grad_features / (np.linalg.norm(grad_features, axis=1, keepdims=True) + 1e-8)
            
            features = alpha * pred_norm + beta * grad_norm
        else:
            features = pred_features
            
        return features

    def _compute_gradient_features(self):
        """Compute meaningful gradient-based features"""
        # Use softmax predictions and targets
        preds = self.train_softmax
        targets = self.train_target
        
        # Create one-hot encoding
        one_hot = np.zeros_like(preds)
        one_hot[np.arange(len(targets)), targets] = 1
        
        # Compute cross-entropy gradients w.r.t. logits
        gradients = preds - one_hot
        
        # Compute gradient norm (informativeness measure)
        grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        
        # Normalize gradients by their norms for direction
        normalized_grads = gradients / (grad_norms + 1e-8)
        
        # Combine gradient direction with magnitude information
        grad_features = np.concatenate([normalized_grads, grad_norms], axis=1)
        
        return grad_features

    def _compute_prediction_features(self):
        """Compute prediction-based features that capture uncertainty and confidence"""
        preds = self.train_softmax
        
        # Max prediction confidence
        max_conf = np.max(preds, axis=1, keepdims=True)
        
        # Entropy (uncertainty measure)
        entropy = -np.sum(preds * np.log(preds + 1e-8), axis=1, keepdims=True)
        
        # Margin (difference between top two predictions)
        sorted_preds = np.sort(preds, axis=1)
        margin = (sorted_preds[:, -1] - sorted_preds[:, -2]).reshape(-1, 1)
        
        # Combine all prediction features
        pred_features = np.concatenate([preds, max_conf, entropy, margin], axis=1)
        
        return pred_features

    def _compute_sample_quality(self):
        """Compute quality scores for each sample"""
        preds = self.train_softmax
        targets = self.train_target
        
        if self.quality_metric == 'loss':
            # Use cross-entropy loss as quality measure
            sample_losses = -np.log(np.maximum(preds[np.arange(len(targets)), targets], 1e-8))
            quality_scores = 1.0 / (1.0 + sample_losses)  # Higher quality = lower loss
            
        elif self.quality_metric == 'uncertainty':
            # Use prediction uncertainty as quality measure
            entropy = -np.sum(preds * np.log(preds + 1e-8), axis=1)
            quality_scores = entropy / np.log(self.args.num_classes)  # Normalized entropy
            
        elif self.quality_metric == 'combined':
            # Combine multiple quality measures
            sample_losses = -np.log(np.maximum(preds[np.arange(len(targets)), targets], 1e-8))
            entropy = -np.sum(preds * np.log(preds + 1e-8), axis=1)
            margin = np.max(preds, axis=1) - np.partition(preds, -2, axis=1)[:, -2]
            
            # Normalize each measure
            loss_norm = 1.0 / (1.0 + sample_losses)
            entropy_norm = entropy / np.log(self.args.num_classes)
            margin_norm = 1.0 - margin  # Lower margin = higher uncertainty = higher quality
            
            # Weighted combination
            quality_scores = 0.4 * loss_norm + 0.4 * entropy_norm + 0.2 * margin_norm
        
        return quality_scores

    def _combined_selection_strategy(self, features, quality_scores, target_size, epoch):
        """Combined selection strategy using uncertainty, diversity, and coverage"""
        
        # Step 1: Uncertainty-based selection
        uncertainty_budget = int(target_size * self.uncertainty_weight)
        uncertainty_indices = self._uncertainty_selection(quality_scores, uncertainty_budget)
        
        # Step 2: Diversity-based selection from remaining samples
        remaining_indices = np.setdiff1d(self.train_indices, uncertainty_indices)
        diversity_budget = int(target_size * self.diversity_weight)
        
        if len(remaining_indices) > 0 and diversity_budget > 0:
            diversity_indices = self._diversity_selection(
                features[remaining_indices], remaining_indices, diversity_budget
            )
        else:
            diversity_indices = np.array([])
        
        # Step 3: Coverage-based selection from remaining samples
        used_indices = np.concatenate([uncertainty_indices, diversity_indices]) if len(diversity_indices) > 0 else uncertainty_indices
        remaining_indices = np.setdiff1d(self.train_indices, used_indices)
        coverage_budget = target_size - len(used_indices)
        
        if len(remaining_indices) > 0 and coverage_budget > 0:
            coverage_indices = self._coverage_selection(
                features[remaining_indices], remaining_indices, coverage_budget
            )
        else:
            coverage_indices = np.array([])
        
        # Combine all selected indices
        all_selected = []
        selection_weights = []
        
        if len(uncertainty_indices) > 0:
            all_selected.extend(uncertainty_indices)
            # Higher weights for uncertain samples
            uncertainty_weights = quality_scores[uncertainty_indices] * 1.5
            selection_weights.extend(uncertainty_weights)
        
        if len(diversity_indices) > 0:
            all_selected.extend(diversity_indices)
            # Standard weights for diverse samples
            diversity_weights = np.ones(len(diversity_indices))
            selection_weights.extend(diversity_weights)
        
        if len(coverage_indices) > 0:
            all_selected.extend(coverage_indices)
            # Lower weights for coverage samples
            coverage_weights = np.ones(len(coverage_indices)) * 0.8
            selection_weights.extend(coverage_weights)
        
        # Normalize weights
        selection_weights = np.array(selection_weights)
        if len(selection_weights) > 0:
            selection_weights = selection_weights / np.sum(selection_weights) * len(selection_weights)
        
        return np.array(all_selected), selection_weights

    def _uncertainty_selection(self, quality_scores, budget):
        """Select most uncertain/informative samples"""
        if budget <= 0:
            return np.array([])
        
        # Select samples with highest uncertainty scores
        uncertain_indices = np.argsort(quality_scores)[-budget:]
        return uncertain_indices

    def _diversity_selection(self, features, candidate_indices, budget):
        """Select diverse samples using a simple greedy approach"""
        if budget <= 0 or len(candidate_indices) == 0:
            return np.array([])
        
        # Normalize features
        normalized_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        selected = []
        remaining = list(range(len(candidate_indices)))
        
        # Select first sample (most representative)
        if remaining:
            similarities = np.dot(normalized_features, normalized_features.T)
            rep_scores = np.sum(similarities, axis=1)
            first_idx = np.argmax(rep_scores)
            selected.append(first_idx)
            remaining.remove(first_idx)
        
        # Greedily select diverse samples
        while len(selected) < budget and remaining:
            best_idx = -1
            best_diversity = -1
            
            for idx in remaining:
                # Compute minimum similarity to already selected samples
                if selected:
                    min_similarity = np.min([
                        np.dot(normalized_features[idx], normalized_features[sel_idx])
                        for sel_idx in selected
                    ])
                    diversity_score = 1.0 - min_similarity
                else:
                    diversity_score = 1.0
                
                if diversity_score > best_diversity:
                    best_diversity = diversity_score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return candidate_indices[selected]

    def _coverage_selection(self, features, candidate_indices, budget):
        """Select samples for coverage using facility location"""
        if budget <= 0 or len(candidate_indices) == 0:
            return np.array([])
        
        try:
            from submodlib import FacilityLocationFunction
            
            # Normalize features
            normalized_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            # Compute similarity matrix
            similarity_matrix = np.dot(normalized_features, normalized_features.T)
            
            # Create facility location object
            fl_obj = FacilityLocationFunction(
                n=len(features),
                mode="dense",
                sijs=similarity_matrix,
                separate_rep=False
            )
            
            # Select using greedy optimization
            selected_items = fl_obj.maximize(
                budget=min(budget, len(candidate_indices)),
                optimizer="LazyGreedy",
                stopIfZeroGain=False,
                stopIfNegativeGain=False,
                verbose=False
            )
            
            selected_indices = [item[0] for item in selected_items]
            return candidate_indices[selected_indices]
            
        except ImportError:
            self.args.logger.warning("SubModLib not available, using random selection for coverage")
            if budget >= len(candidate_indices):
                return candidate_indices
            else:
                return np.random.choice(candidate_indices, size=budget, replace=False)

    def _stratified_random_selection(self, target_size):
        """Stratified random selection that maintains class balance"""
        targets = self.train_target
        unique_classes = np.unique(targets)
        
        selected_indices = []
        for cls in unique_classes:
            class_indices = np.where(targets == cls)[0]
            class_fraction = len(class_indices) / len(targets)
            class_target_size = int(target_size * class_fraction)
            
            if class_target_size > len(class_indices):
                class_selected = class_indices
            else:
                class_selected = np.random.choice(class_indices, size=class_target_size, replace=False)
            
            selected_indices.extend(class_selected)
        
        # If we haven't selected enough samples, add more randomly
        if len(selected_indices) < target_size:
            remaining_indices = np.setdiff1d(self.train_indices, selected_indices)
            additional_needed = target_size - len(selected_indices)
            if len(remaining_indices) >= additional_needed:
                additional_selected = np.random.choice(remaining_indices, size=additional_needed, replace=False)
                selected_indices.extend(additional_selected)
        
        selected_indices = np.array(selected_indices[:target_size])
        weights = np.ones(len(selected_indices))
        
        return selected_indices, weights

    def _get_train_output_efficient(self, indices=None):
        """Efficiently compute model outputs with caching"""
        # Use the same implementation as the parent class
        super()._get_train_output_efficient(indices) 
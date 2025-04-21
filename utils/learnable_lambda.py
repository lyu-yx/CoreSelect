import torch
import numpy as np
import wandb


class LearnableLambda:
    """
    Implements a truly learnable lambda (λ) parameter for DPP weight optimization.
    
    This class optimizes the diversity-coverage trade-off parameter through gradient-based
    learning on a meta-objective (validation performance) and maintains per-class
    parameters when beneficial.
    """
    
    def __init__(
        self,
        num_classes,
        initial_value=0.5,
        min_value=0.2,
        max_value=0.8,
        meta_lr=0.01,
        use_per_class=True,
        device='cuda'
    ):
        """
        Initialize the learnable lambda parameter.
        
        Args:
            num_classes: Number of classes in the dataset
            initial_value: Starting value for lambda
            min_value: Minimum allowed value for lambda
            max_value: Maximum allowed value for lambda
            meta_lr: Learning rate for the meta-optimizer
            use_per_class: Whether to use per-class lambda values
            device: Device to store parameters on
        """
        self.num_classes = num_classes
        self.min_value = min_value
        self.max_value = max_value
        self.meta_lr = meta_lr
        self.use_per_class = use_per_class
        self.device = device
        self.requires_meta_update = False
        self.meta_update_frequency = 10  # Update every 10 batches by default
        self.update_counter = 0
        
        # Initialize the raw learnable parameter(s)
        if use_per_class:
            # One lambda per class - initialize with the same value
            self.raw_lambda = torch.nn.Parameter(
                torch.ones(num_classes) * self._inverse_transform(initial_value),
                requires_grad=True
            )
        else:
            # Single global lambda
            self.raw_lambda = torch.nn.Parameter(
                torch.tensor([self._inverse_transform(initial_value)]),
                requires_grad=True
            )
            
        # Create optimizer for the parameter
        self.meta_optimizer = torch.optim.Adam([self.raw_lambda], lr=meta_lr)
        
        # Performance tracking for meta-optimization
        self.val_performances = []
        self.train_performances = []
        
        # Cache for faster lookup
        self.cached_values = {}
        self.cached_epoch = -1
        
    def _inverse_transform(self, normalized_value):
        """Convert from [min_value, max_value] range to unbounded space for optimization"""
        # Using logit transformation: log(p/(1-p))
        normalized = (normalized_value - self.min_value) / (self.max_value - self.min_value)
        # Clip to avoid numerical issues
        normalized = np.clip(normalized, 0.001, 0.999)
        return np.log(normalized / (1.0 - normalized))
    
    def _transform(self, raw_value):
        """Convert from unbounded optimization space to [min_value, max_value] range"""
        # Using sigmoid transformation: 1/(1+e^(-x))
        # Clip raw values for numerical stability
        clipped_raw = torch.clamp(raw_value, -10.0, 10.0) 
        sigmoid = 1.0 / (1.0 + torch.exp(-clipped_raw))
        return self.min_value + sigmoid * (self.max_value - self.min_value)
    
    def get_value(self, class_idx=None, epoch=None):
        """
        Get the current value of lambda, possibly class-specific.
        
        Args:
            class_idx: Class index for per-class lambda (optional)
            epoch: Current epoch for caching purposes
            
        Returns:
            Current lambda value (float)
        """
        # Check cache first for efficiency
        cache_key = (epoch, class_idx)
        if cache_key in self.cached_values:
            return self.cached_values[cache_key]
            
        # Get appropriate raw value
        if self.use_per_class and class_idx is not None:
            raw_value = self.raw_lambda[class_idx]
        else:
            raw_value = self.raw_lambda[0]
            
        # Transform to proper range
        transformed = self._transform(raw_value).item()
        
        # Cache for later use
        if epoch is not None:
            # Maintain a reasonable cache size - clear old entries if needed
            if len(self.cached_values) > 1000:  # Arbitrary threshold
                # Keep only the most recent epoch's values
                current_epoch_keys = [k for k in self.cached_values.keys() if k[0] == epoch]
                self.cached_values = {k: self.cached_values[k] for k in current_epoch_keys}
            
            self.cached_values[cache_key] = transformed
            self.cached_epoch = epoch
            
        return transformed
        
    def get_all_values(self):
        """Get all lambda values as a numpy array"""
        return self._transform(self.raw_lambda).detach().cpu().numpy()
    
    def meta_backward(self, val_loss):
        """
        Perform meta-backward pass to update lambda based on validation loss.
        
        Args:
            val_loss: Validation loss to minimize
        """
        if not self.requires_meta_update:
            return
            
        # Zero gradients
        self.meta_optimizer.zero_grad()
        
        # Use validation loss as meta-objective
        val_loss.backward(retain_graph=True)
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_([self.raw_lambda], 1.0)
        
        # Update parameters
        self.meta_optimizer.step()
        
        # Reset meta update flag
        self.requires_meta_update = False
    
    def log_to_wandb(self, epoch, additional_data=None):
        """
        Log current lambda values to wandb
        
        Args:
            epoch: Current epoch
            additional_data: Any additional data to log
        """
        if not wandb.run:
            return
            
        log_data = {'epoch': epoch}
        
        # Log global or per-class lambda values
        if self.use_per_class:
            values = self.get_all_values()
            for i, val in enumerate(values):
                log_data[f'lambda_class_{i}'] = val
            log_data['lambda_mean'] = values.mean()
            log_data['lambda_std'] = values.std()
        else:
            log_data['lambda'] = self.get_value()
            
        # Add any additional data
        if additional_data:
            log_data.update(additional_data)
            
        # Log to wandb
        wandb.log(log_data)
    
    def update_with_performance(self, train_perf, val_perf, epoch):
        """
        Update lambda based on training and validation performance trends.
        This implements a meta-learning approach that adjusts lambda based on
        how it affects generalization.
        
        Args:
            train_perf: Training performance metric (e.g., accuracy)
            val_perf: Validation performance metric
            epoch: Current epoch
        """
        # Track performance
        self.train_performances.append(train_perf)
        self.val_performances.append(val_perf)
        
        # Need at least a few data points to detect trends
        if len(self.val_performances) < 3:
            return
            
        # Clear old cache if epoch changed
        if epoch != self.cached_epoch:
            self.cached_values = {}
        
        # Update counter
        self.update_counter += 1
        
        # Only update periodically
        if self.update_counter % self.meta_update_frequency != 0:
            return
            
        # Calculate generalization gap trend
        recent_gaps = [t - v for t, v in zip(self.train_performances[-3:], self.val_performances[-3:])]
        gap_increasing = recent_gaps[-1] > recent_gaps[0]
        
        # Calculate validation performance trend
        val_trend = self.val_performances[-1] - self.val_performances[-3]
        
        # Logic for updating lambda:
        # 1. If gap is increasing, increase lambda (more diversity needed)
        # 2. If gap is decreasing but val performance is also decreasing, 
        #    increase lambda (likely underfitting)
        # 3. Otherwise, decrease lambda (focus on coverage)
        
        adjustment = 0.05  # Small adjustment step
        
        if gap_increasing:
            # Generalization gap increasing - need more diversity
            update_direction = adjustment
        elif val_trend < 0:
            # Val performance decreasing - need more diversity
            update_direction = adjustment
        else:
            # Healthy training - can reduce diversity
            update_direction = -adjustment
            
        # Apply update to raw parameter with proper gradient
        with torch.no_grad():
            current_values = self._transform(self.raw_lambda)
            new_values = torch.clamp(
                current_values + update_direction,
                self.min_value, 
                self.max_value
            )
            # Convert back to raw space
            for i in range(len(self.raw_lambda)):
                self.raw_lambda[i] = self._inverse_transform(new_values[i].item())
        
        # Set flag for next backward pass
        self.requires_meta_update = True
        
        # Log the update
        direction_name = "increased" if update_direction > 0 else "decreased"
        print(f"Meta-updated lambda {direction_name} by {abs(update_direction):.3f} based on performance")
        
        # Log to wandb
        self.log_to_wandb(epoch, {
            'lambda_update_direction': update_direction,
            'generalization_gap': recent_gaps[-1],
            'gap_trend': recent_gaps[-1] - recent_gaps[0],
            'val_performance_trend': val_trend
        })
from utils import Adahessian
from datasets.subset import get_coreset
from .subset_trainer import *
from time import time


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
        self.random_sets = np.array([])

        self.num_checking = 0

        self.gradient_approx_optimizer = Adahessian(self.model.parameters())

        self.loss_watch = np.ones((self.args.watch_interval, len(self.train_dataset))) * -1

        self.approx_time = AverageMeter()
        self.compare_time = AverageMeter()
        self.similarity_time = AverageMeter()

    def _train_epoch(self, epoch: int):
        """
        Train the model for one epoch
        :param epoch: current epoch
        """
        self.model.train()
        self._reset_metrics()

        lr = self.lr_scheduler.get_last_lr()[0]
        self.args.logger.info(f"Epoch {epoch} LR {lr:.6f}")

        for training_step in range(self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)):

            # if (training_step > self.reset_step) and ((training_step - self.reset_step) % self.args.check_interval == 0):
            #     self._check_approx_error(epoch, training_step)

            # if epoch >= self.args.drop_after and (epoch % self.args.drop_interval == 0) and training_step == self.steps_per_epoch * epoch: # drop detrimental data at the begining of target epoch
            #     subset = self._select_subset_drop_detrimental(epoch, training_step)
            #     keep = np.where(self.times_selected[subset] == epoch)[0]
            #     subset = subset[keep]
            #     self._update_train_loader_and_weights()

            if training_step % self.steps_per_epoch == 0 and training_step >= self.steps_per_epoch:
                self._select_subset_drop_detrimental(epoch, training_step)
                self._update_train_loader_and_weights()
                self.train_iter = iter(self.train_loader)
                self._get_quadratic_approximation(epoch, training_step)
                
            elif training_step == 0:
                self.train_iter = iter(self.train_loader)

            data_start = time.time()
            try:
                batch = next(self.train_iter)
            except StopIteration:
                if self.args.cache_dataset and self.args.clean_cache_iteration:
                    self.train_dataset.clean()
                    self._update_train_loader_and_weights()
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            data, target, data_idx = batch
            data, target = data.to(self.args.device), target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            loss, train_acc = self._forward_and_backward(data, target, data_idx)

            data_start = time.time()

            if self.args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "training_step": training_step,
                    "train_loss": loss.item(),
                    "train_acc": train_acc})


    def _forward_and_backward(self, data, target, data_idx):
        self.optimizer.zero_grad()

        # train model with the current batch and record forward and backward time
        forward_start = time.time()
        output = self.model(data)
        forward_time = time.time() - forward_start
        self.batch_forward_time.update(forward_time)

        loss = self.train_criterion(output, target)
        loss = (loss * self.train_weights[data_idx]).mean()

        lr = self.lr_scheduler.get_last_lr()[0]
        if lr > 0:
            # compute the parameter change delta
            self.model.zero_grad()
            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_current, _, _ = self.gradient_approx_optimizer.step(momentum=False)                   
            self.delta -= lr * gf_current

        backward_start = time.time()
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.batch_backward_time.update(backward_time)

        # update training loss and accuracy
        train_acc = (output.argmax(dim=1) == target).float().mean().item()
        self.train_loss.update(loss.item(), data.size(0))
        self.train_acc.update(train_acc, data.size(0))

        return loss, train_acc


    def _get_quadratic_approximation(self, epoch: int, training_step: int):
        """
        Compute the quadratic approximation of the loss function
        :param epoch: current epoch
        :param training_step: current training step
        """

        if self.args.approx_with_coreset:
            # Update the second-order approximation with the coreset
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.subset),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )
        else:
            # Update the second-order approximation with random subsets
            approx_loader = DataLoader(
                Subset(self.train_dataset, indices=self.random_sets),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )

        approx_start = time.time()
        curvature_norm = AverageMeter()
        self.start_loss = AverageMeter()
        for approx_batch, (input, target, idx) in enumerate(approx_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target

            # compute output
            output = self.model(input_var)
                
            if self.args.approx_with_coreset:
                loss = self.train_criterion(output, target_var)
                batch_weight = self.train_weights[idx.long()]
                loss = (loss * batch_weight).mean()
            else:
                loss = self.val_criterion(output, target_var)
            self.model.zero_grad()

            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(momentum=True)

            if approx_batch == 0:
                self.gf = gf_tmp * len(idx)
                self.ggf = ggf_tmp * len(idx)
                self.ggf_moment = ggf_tmp_moment * len(idx)
            else:
                self.gf += gf_tmp * len(idx)
                self.ggf += ggf_tmp * len(idx)
                self.ggf_moment += ggf_tmp_moment * len(idx)

            curvature_norm.update(ggf_tmp_moment.norm())
            self.start_loss.update(loss.item(), input.size(0))

        approx_time = time.time() - approx_start
        self.approx_time.update(approx_time)

        self.gf /= len(approx_loader.dataset)
        self.ggf /= len(approx_loader.dataset)
        self.ggf_moment /= len(approx_loader.dataset)
        self.delta = 0

        gff_norm = curvature_norm.avg
        self.start_loss = self.start_loss.avg
        if self.args.approx_moment:
            self.ggf = self.ggf_moment

        if training_step == self.steps_per_epoch:
            self.init_curvature_norm = gff_norm 
        else:
            self.args.check_interval = int(torch.ceil(self.init_curvature_norm / gff_norm * self.args.interval_mul))
            self.args.num_minibatch_coreset = min(self.args.check_interval * self.args.batch_num_mul, self.steps_per_epoch)
        self.args.logger.info(f"Checking interval {self.args.check_interval}. Number of minibatch coresets {self.args.num_minibatch_coreset}")
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'training_step': training_step,
                'ggf_norm': gff_norm,
                'check_interval': self.args.check_interval,
                'num_minibatch_coreset': self.args.num_minibatch_coreset})

    def _check_approx_error(self, epoch:int, training_step: int) -> torch.Tensor:
        """
        Check the approximation error of the current batch
        :param epoch: current epoch
        :param training_step: current training step
        """
        self.num_checking += 1
        start_compare = time.time()
        self._get_train_output()
        true_loss = self.val_criterion(
            torch.from_numpy(self.train_output[self.random_sets]), 
            torch.from_numpy(self.train_target[self.random_sets])
            )
        
        delta_norm = torch.norm(self.delta)

        approx_loss = torch.matmul(self.delta, self.gf) + self.start_loss
        approx_loss += 1 / 2 * torch.matmul(self.delta * self.ggf, self.delta)

        loss_diff = abs(true_loss - approx_loss.item())
        thresh = self.args.check_thresh_factor * true_loss

        log_str = f"Iter {training_step} loss difference {loss_diff:.3f} threshold {thresh:.3f} True loss {true_loss:.3f} Approx loss {approx_loss.item():.3f} Delta norm {delta_norm:.3f}"
            
        if loss_diff > thresh:
            self.reset_step = training_step
            log_str += f" is larger than threshold {thresh:.3f}. "
        self.args.logger.info(log_str)

        compare_time = time.time() - start_compare
        self.compare_time.update(compare_time)

        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'training_step': training_step,
                'loss_diff': loss_diff, 
                'loss_thresh': thresh,
                'delta_norm': delta_norm,
                'num_checking': self.num_checking})
            
    def _drop_learned_data(self, epoch: int, training_step: int, indices: np.ndarray):
        """
        Drop the learned data points
        :param epoch: current epoch
        :param training_step: current training step
        :param indices: indices of the data points that have valid predictions
        """
        
        self.loss_watch[epoch % self.args.watch_interval, indices] = self.train_criterion(
            torch.from_numpy(self.train_output[indices]), torch.from_numpy(self.train_target[indices]).long()).numpy()
                        
        if ((epoch+1) % self.args.drop_interval == 0):
            # 1. identifies data points that have large loss values or Untracked instance.
            # 2. selects a subset of the data points above selected set.
            order_ = np.where(np.sum(self.loss_watch>self.args.drop_thresh, axis=0)>0)[0]
            unselected = np.where(np.sum(self.loss_watch>=0, axis=0)==0)[0]
            order_ = np.concatenate([order_, unselected])

            order = []
            per_class_size = int(np.ceil(self.args.random_subset_size * self.args.train_size / self.args.num_classes))
            for c in np.unique(self.train_target):
                class_indices_new = np.intersect1d(np.where(self.train_target == c)[0], order_)
                if len(class_indices_new) > per_class_size:
                    order.append(class_indices_new)
                else:
                    class_indices = np.intersect1d(np.where(self.train_target == c)[0], self.train_indices)
                    order.append(class_indices)
            order = np.concatenate(order)
            
            if len(order) > self.args.min_train_size:
                self.train_indices = order

            if self.args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'forgettable_train': len(self.train_indices)})
                


    def _drop_detrimental_data(self, epoch: int, training_step: int, indices: np.ndarray, preds):
        """
        Drop the detrimental data points using clustering-based filtering.
        
        :param epoch: current epoch
        :param training_step: current training step
        :param indices: indices of the data points that have valid predictions
        :param preds: predictions corresponding to those indices
        :return: updated indices after dropping detrimental data.
        """
        if len(indices) == 0:
            return indices
            
        N = len(indices)
        # Ensure we don't try to sample more than available
        class_counts = np.bincount(self.train_target[indices])
        min_class_count = np.min(class_counts[class_counts > 0])
        B = min(self.args.detrimental_sampled, N, min_class_count * self.args.num_classes)
        
        try:
            # Step 1: Initial clustering to group instances with similar gradient behavior
            subset, subset_weights, _, _, cluster_ = get_coreset(
                preds,
                self.train_target[indices],
                N,
                B,
                self.args.num_classes,
            )
            # rest of processing
        except Exception as e:
            self.args.logger.warning(f"Error in coreset selection: {e}. Using original indices.")
            return indices
        
        # Map the selected subset back to global indices, the subset number should be B
        subset = indices[subset]
        
        # Create a cluster mapping for the full dataset
        cluster = -np.ones(len(self.train_target), dtype=int)
        cluster[indices] = cluster_
        
        # Identify clusters to keep based on the weight threshold
        # Dynamic threshold based on desired drop percentage
        if hasattr(self.args, 'target_drop_percentage'):
            # Sort weights and find threshold that gives target percentage
            sorted_weights = np.sort(subset_weights)
            target_idx = int(len(sorted_weights) * (1 - self.args.target_drop_percentage/100))
            target_idx = max(0, min(target_idx, len(sorted_weights)-1))  # Ensure valid index
            dynamic_threshold = sorted_weights[target_idx]
            
            # Use the dynamic threshold, but don't go below the minimum threshold
            actual_threshold = max(dynamic_threshold, self.args.cluster_thresh)
            valid_clusters = np.where(subset_weights > actual_threshold)[0]
            
            self.args.logger.info(f"Using dynamic threshold: {actual_threshold:.4f} "
                                 f"(target: {self.args.target_drop_percentage}%)")
        else:
            # Use fixed threshold as before
            valid_clusters = np.where(subset_weights > self.args.cluster_thresh)[0]
            
        # Only apply filtering after certain epoch
        if epoch >= self.args.drop_after:
            # Keep only data points that belong to the selected clusters
            keep_mask = np.isin(cluster, valid_clusters)
            keep_indices = np.where(keep_mask)[0]
            
            # Safety check: ensure we're not dropping too many points
            if len(keep_indices) < self.args.min_batch_size:
                self.args.logger.warning(
                    f"Too few samples after detrimental drop: {len(keep_indices)}. "
                    f"Using original indices instead."
                )
                keep_indices = indices
        else:
            keep_indices = indices
    
        # Ensure we only return valid points that were in the original indices
        updated_indices = np.intersect1d(indices, keep_indices)
        
        drop_percentage = 100 * (1 - len(updated_indices) / len(indices))
        self.args.logger.info(
            f"Epoch {epoch}: Kept {len(updated_indices)}/{len(indices)} points after detrimental filtering "
            f"({drop_percentage:.2f}% dropped)"
        )
        
        # Log detailed metrics about cluster weights
        if self.args.use_wandb:
            metrics = {
                'epoch': epoch,
                'training_step': training_step,
                'detrimental_drop_count': len(updated_indices),
                'detrimental_drop_percentage': drop_percentage
            }
            
            if len(subset_weights) > 0:
                metrics.update({
                    'min_cluster_weight': np.min(subset_weights),
                    'max_cluster_weight': np.max(subset_weights),
                    'mean_cluster_weight': np.mean(subset_weights),
                    'cluster_threshold': self.args.cluster_thresh,
                    'valid_clusters_count': len(valid_clusters),
                    'total_clusters_count': len(subset_weights)
                })
                
            wandb.log(metrics)
        
        return updated_indices

    def _select_random_set(self) -> np.ndarray:
        indices = []
        for c in np.unique(self.train_target):
            class_indices = np.intersect1d(np.where(self.train_target == c)[0], self.train_indices)
            indices_per_class = np.random.choice(class_indices, size=int(np.ceil(self.args.random_subset_size * self.args.train_size / self.args.num_classes)), replace=False)
            indices.append(indices_per_class)
        indices = np.concatenate(indices)
        return indices

    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data - ensure this uses the proper gradient flow for 
        aligning with the theoretical framework of:
        F(S) = f(S) + λD(S)
        where f(S) is the coverage function and D(S) is the diversity function.
        """
        super()._select_subset(epoch, training_step)
        
        # get random subsets
        self.random_sets = []
        self.subset = []
        self.subset_weights = []
        
        for _ in range(self.args.num_minibatch_coreset):
            # get a random subset of the data
            random_subset = self._select_random_set()
            self.random_sets.append(random_subset)
            
        self.train_val_loader = DataLoader(
            Subset(self.train_dataset, indices=np.concatenate(self.random_sets)),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        
        self._get_train_output()
        
        # drop the learned data points
        if self.args.drop_learned:
            self._drop_learned_data(epoch, training_step, np.concatenate(self.random_sets))
            
        for random_set in self.random_sets:
            preds = self.train_softmax[random_set]
            print(f"Epoch [{epoch}] [Greedy], pred size: {np.shape(preds)}")
            
            if np.shape(preds)[-1] == self.args.num_classes:
                # This is the gradient of cross-entropy loss w.r.t. logits: ∇L(x,θ) = softmax(logits) - one_hot
                # This aligns with the theory where we use gradients to measure sample importance
                preds -= np.eye(self.args.num_classes)[self.train_target[random_set]]
                
                # The subset selection objective implicitly optimizes for:
                # f(S) = coverage of gradient space (via facility location)
                subset, weight, _, similarity_time = self.subset_generator.generate_subset(
                    preds=preds,  # These are the gradients
                    epoch=epoch,
                    B=self.args.batch_size,
                    idx=random_set,
                    targets=self.train_target,
                    use_submodlib=(self.args.smtk==0),
                )
                self.similarity_time.update(similarity_time)
                self.subset.append(subset)
                self.subset_weights.append(weight)
                
        self.subset = np.concatenate(self.subset)
        self.subset_weights = np.concatenate(self.subset_weights)
        self.random_sets = np.concatenate(self.random_sets)

    def _select_subset_drop_detrimental(self, epoch: int, training_step: int):
        """
        Select a subset of the data, dropping both learned and detrimental data points.
        Incorporates DPP-based joint objective for better diversity and coverage.
        
        Key theoretical components:
        1. Gradient-based filtering to remove detrimental points
        2. Joint objective F(S) = f(S) + λD(S) for final selection
        """

        # ----------------------------
        # 1. Select random subsets
        # ----------------------------
        self.random_sets = []
        self.subset = []
        self.subset_weights = []
        
        # Track selection for metrics (originally done in parent method)
        self.num_selection += 1
        
        # Log selection event to wandb (originally done in parent method)
        if self.args.use_wandb:
            wandb.log({"epoch": epoch, "training_step": training_step, "num_selection": self.num_selection})
            
        # Handle dataset caching if needed (originally done in parent method)
        if self.args.cache_dataset:
            self.train_dataset.clean()
            self.train_dataset.cache()
            
        # Continue with the existing implementation
        for _ in range(self.args.num_minibatch_coreset):
            # Select a random subset of global indices.
            random_subset = self._select_random_set()
            self.random_sets.append(random_subset)
            
        # Combine all indices for global learned data dropping.
        combined_random_sets = np.concatenate(self.random_sets)
        
        # ----------------------------
        # 2. Drop learned data points globally
        # ----------------------------
        if self.args.drop_learned:
            # This updates self.train_indices based on loss values.
            time_start = time.time()
            self._drop_learned_data(epoch, training_step, combined_random_sets)
            time_end = time.time()
            self.args.logger.info(f"dropping learned data: {time_end - time_start:.2f} seconds")
            
        # Filter each random set so they contain only indices in self.train_indices.
        self.random_sets = [np.intersect1d(rs, self.train_indices) for rs in self.random_sets]
        
        # Update the DataLoader based on the filtered indices.
        self.train_val_loader = DataLoader(
            Subset(self.train_dataset, indices=np.concatenate(self.random_sets)),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )


        time_start = time.time()
        self._get_train_output()  # Update all random_set gradient.
        time_end = time.time()
        self.args.logger.info(f"_get_train_output: {time_end - time_start:.2f} seconds")
        # ----------------------------
        # 3. Process each random subset individually
        # ----------------------------
        updated_random_sets = []
        processed_subsets = []
        processed_weights = []
        
        for orig_random_set in self.random_sets:
            # Make a local copy so that we do not affect the global self.random_sets.
            local_set = orig_random_set.copy()
            
            # Get predictions for the local set.
            if len(local_set) > 0:
                # Calculate gradients once for both detrimental dropping and selection
                preds = self.train_softmax[local_set]
                print(f"Epoch [{epoch}] [Greedy], initial pred shape: {np.shape(preds)}")
                
                # First apply detrimental dropping using the predictions
                if self.args.drop_detrimental and len(local_set) > 0:
                    original_size = len(local_set)
                    # This is where we identify and remove points that hurt training
                    local_set = self._drop_detrimental_data(epoch, training_step, local_set, preds)
                    # Safety check - if we dropped all points,
                    # revert to original set
                    if len(local_set) == 0:
                        self.args.logger.warning(
                            f"All points were dropped as detrimental. Using original set."
                        )
                        local_set = orig_random_set.copy()
                    
                    # Recompute predictions using the updated local_set.
                    preds = self.train_softmax[local_set]
                    print(f"Epoch [{epoch}] [Greedy], after detrimental drop: {len(local_set)}/{original_size} points remain")
                
                # Skip subset generation if local_set is empty
                if len(local_set) > 0:
                    # Calculate gradient approximation for the joint objective
                    if np.shape(preds)[-1] == self.args.num_classes:
                        one_hot_labels = np.eye(self.args.num_classes)[self.train_target[local_set]]
                        
                        # Compute gradients as in CREST: g = softmax(logits) - one_hot(true_labels)
                        if one_hot_labels.shape == preds.shape:
                            # This calculates the gradient of the cross-entropy loss with respect to the logits
                            gradients = preds - one_hot_labels
                        else:
                            self.args.logger.error(
                                f"Shape mismatch: preds {preds.shape} vs one_hot {one_hot_labels.shape}. "
                                f"Skipping gradient calculation."
                            )
                            gradients = preds  # Fall back to using softmax outputs if shape mismatch
                    
                    # Apply the joint objective F(S) = f(S) + λD(S)
                    if hasattr(self.subset_generator, 'generate_mixed_subset'):
                        # Cache label and softmax data to avoid repeated access
                        local_labels = self.train_target[local_set]
                        local_softmax = self.train_softmax[local_set]
                        
                        # Features should be the computed gradients for theoretical alignment
                        # Note: We reuse the gradients already computed above rather than recalculating
                        features = gradients
                        
                        
                        time_start = time.time()
                        # This explicitly optimizes the joint objective from the theory
                        subset_indices, weights = self.subset_generator.generate_mixed_subset(
                            features=features,  # Gradient features for diversity kernel
                            labels=local_labels,
                            softmax_preds=local_softmax,
                            subset_size=min(self.args.batch_size, len(local_set)),
                            dpp_weight=self.args.dpp_weight,  # This is λ in F(S) = f(S) + λD(S)
                            submod_weight=1.0 - self.args.dpp_weight,
                            selection_method="mixed"
                        )
                        time_end = time.time()
                        self.args.logger.info(f"generate_mixed_subset: {time_end - time_start:.2f} seconds")
                        # Map to global indices
                        subset = local_set[subset_indices]
                        self.args.logger.info(f"Using joint DPP+coverage selection: {len(subset)} points selected")
                        similarity_time = 0  # Not measured for this path
                    else:
                        # Fall back to legacy selection method
                        subset, weights, _, similarity_time = self.subset_generator.generate_subset(
                            preds=gradients,  # Use computed gradients instead of preds
                            epoch=epoch,
                            B=min(self.args.batch_size, len(local_set)),  # Don't request more than available
                            idx=local_set,
                            targets=self.train_target,
                            use_submodlib=(self.args.smtk == 0),
                        )
                    
                    self.similarity_time.update(similarity_time)
                    processed_subsets.append(subset)
                    processed_weights.append(weights)
            
            # Save the updated local set.
            updated_random_sets.append(local_set)
        
        # Update global variables after processing, handle empty case
        if updated_random_sets and any(len(rs) > 0 for rs in updated_random_sets):
            self.random_sets = np.concatenate([rs for rs in updated_random_sets if len(rs) > 0])
        else:
            self.random_sets = np.array([], dtype=int)
            
        if processed_subsets:
            self.subset = np.concatenate(processed_subsets)
            self.subset_weights = np.concatenate(processed_weights)
        else:
            self.subset = np.array([], dtype=int)
            self.subset_weights = np.array([], dtype=float)
            
        # Log summary stats
        total_remaining = len(self.subset)
        self.args.logger.info(f"Epoch {epoch}: Selected final subset of {total_remaining} instances")
        
        if self.args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'training_step': training_step,
                'final_subset_size': total_remaining,
                'joint_dpp_applied': hasattr(self.subset_generator, 'generate_mixed_subset')
            })




# a trainer class that inherits from BaseTrainer and only trains on a subset of the data selected at the beginning of every epoch
from .base_trainer import *
from torch.utils.data import Subset, DataLoader
from datasets import SubsetGenerator

class SubsetTrainer(BaseTrainer):
    def __init__(
        self, 
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        self.train_target = np.array(self.train_dataset.dataset.targets)
        self.subset_generator = SubsetGenerator(greedy=(args.selection_method!="rand"), smtk=args.smtk)

        self.num_selection = 0

    def _update_train_loader_and_weights(self):
        self.args.logger.info("Updating train loader and weights with subset of size {}".format(len(self.subset)))
        self.train_loader = DataLoader(
            Subset(self.train_dataset, self.subset),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.train_val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.train_weights = np.zeros(len(self.train_dataset))
        self.subset_weights = self.subset_weights / np.sum(self.subset_weights) * len(self.subset)
        self.train_weights[self.subset] = self.subset_weights
        self.train_weights = torch.from_numpy(self.train_weights).float().to(self.args.device)


    def _train_epoch(self, epoch):
        # Only select a subset of the data if needed
        # Check if we need to select a new subset for this epoch
        need_to_select = True
        if hasattr(self, '_needs_subset_refresh'):
            # Use the flag set by the BaseTrainer
            need_to_select = self._needs_subset_refresh
        
        if need_to_select:
            self.args.logger.info(f"Epoch {epoch}: Selecting new subset")
            self._select_subset(epoch, len(self.train_loader) * epoch)
            self._update_train_loader_and_weights()
        else:
            self.args.logger.info(f"Epoch {epoch}: Reusing existing subset")

        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        # use tqdm to display a smart progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout)
        for batch_idx, (data, target, data_idx) in pbar:

            # load data to device and record data loading time
            data, target = data.to(self.args.device), target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            self.optimizer.zero_grad()

            # train model with the current batch and record forward and backward time
            loss, train_acc = self._forward_and_backward(data, target, data_idx)

            data_start = time.time()

            # update progress bar
            pbar.set_description(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    self.args.epochs,
                    batch_idx * self.args.batch_size + len(data),
                    len(self.train_loader.dataset),
                    100.0 * (batch_idx+1) / len(self.train_loader),
                    loss.item(),
                    train_acc,
                )
            )

        if self.args.cache_dataset and self.args.clean_cache_iteration:
            self.train_dataset.clean()
            self._update_train_loader_and_weights()

    def _get_train_output(self, indices=None):
        """
        Evaluate the model on the training set (or subset) and record the output and softmax
        
        Args:
            indices: If provided, only compute outputs for these indices.
                    Otherwise compute for the entire dataset.
        """
        self.model.eval()

        if indices is None:
            # Initialize arrays for the entire dataset
            if not hasattr(self, 'train_output') or self.train_output.shape[0] != len(self.train_dataset):
                self.train_output = np.zeros((len(self.train_dataset), self.args.num_classes))
                self.train_softmax = np.zeros((len(self.train_dataset), self.args.num_classes))
                
            # Evaluate on all data points via train_val_loader
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )
        else:
            # Initialize arrays if they don't exist
            if not hasattr(self, 'train_output'):
                self.train_output = np.zeros((len(self.train_dataset), self.args.num_classes))
                self.train_softmax = np.zeros((len(self.train_dataset), self.args.num_classes))
                
            # Evaluate only on the specified indices
            loader = DataLoader(
                Subset(self.train_dataset, indices),
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True
            )

        with torch.no_grad():
            for _, (data, _, data_idx) in enumerate(loader):
                data = data.to(self.args.device)

                output = self.model(data)

                self.train_output[data_idx] = output.cpu().numpy()
                self.train_softmax[data_idx] = output.softmax(dim=1).cpu().numpy()

        self.model.train()


    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data
        """
        self.num_selection += 1

        if self.args.use_wandb:
            wandb.log({"epoch": epoch, "training_step": training_step, "num_selection": self.num_selection})

        if self.args.cache_dataset:
            self.train_dataset.clean()
            self.train_dataset.cache()
        pass

    def _get_num_selection(self):
        return self.num_selection

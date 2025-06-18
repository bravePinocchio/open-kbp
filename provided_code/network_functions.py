import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from provided_code.network_architectures import DefineDoseFromCT
from provided_code.data_loader import DataLoader as KerasDataLoader
from provided_code.utils import get_paths, sparse_vector_function

class TorchDataLoader:
    """Wrapper to adapt Keras-style data loader to PyTorch"""
    def __init__(self, keras_data_loader):
        self.keras_data_loader = keras_data_loader
        self.data_shapes = keras_data_loader.data_shapes
        self.full_roi_list = keras_data_loader.full_roi_list
        
    def set_mode(self, mode):
        self.keras_data_loader.set_mode(mode)
        
    def shuffle_data(self):
        self.keras_data_loader.shuffle_data()
        
    def get_batches(self):
        for batch in self.keras_data_loader.get_batches():
            yield TorchBatch(batch)

class TorchBatch:
    """Convert Keras batch to PyTorch tensors"""
    def __init__(self, keras_batch):
        
        self.ct = torch.from_numpy(keras_batch.ct).float()
        self.structure_masks = torch.from_numpy(keras_batch.structure_masks).float()
        self.dose = torch.from_numpy(keras_batch.dose).float() if keras_batch.dose.any() !=None  else None
        self.possible_dose_mask = torch.from_numpy(keras_batch.possible_dose_mask).float()
        self.patient_list = keras_batch.patient_list

class PredictionModel(DefineDoseFromCT):
    def __init__(self, data_loader: KerasDataLoader, results_patent_path: Path, model_name: str, stage: str) -> None:
        """
        :param data_loader: An object that loads batches of image data
        :param results_patent_path: The path at which all results and generated models will be saved
        :param model_name: The name of your model, used when saving and loading data
        :param stage: Identify stage of model development (train, validation, test)
        """
        # Initialize the parent class with torch optimizer
        super().__init__(
            data_shapes=data_loader.data_shapes,
            initial_number_of_filters=64,
            filter_size=(4, 4, 4),
            stride_size=(2, 2, 2),
            gen_optimizer=lambda params: Adam(params, lr=0.0002, betas=(0.5, 0.999)),
        )

        # set attributes for data shape from data loader
        self.generator = None
        self.model_name = model_name
        self.data_loader = TorchDataLoader(data_loader)  # Wrap the Keras data loader
        self.full_roi_list = data_loader.full_roi_list

        # Define training parameters
        self.current_epoch = 0
        self.last_epoch = 200
        self.criterion = torch.nn.L1Loss()  # MAE loss

        # Make directories for data and models
        model_results_path = results_patent_path / model_name
        self.model_dir = model_results_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir = model_results_path / f"{stage}-predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        # Make template for model path (using .pt extension for PyTorch)
        self.model_path_template = self.model_dir / "epoch_"

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_model(self, epochs: int = 200, save_frequency: int = 5, keep_model_history: int = 2) -> None:
        """
        :param epochs: the number of epochs the model will be trained over
        :param save_frequency: how often the model will be saved
        :param keep_model_history: how many models are kept on a rolling basis
        """
        self._set_epoch_start()
        self.last_epoch = epochs
        self.initialize_networks()
        
        if self.current_epoch == epochs:
            print(f"The model has already been trained for {epochs}, so no more training will be done.")
            return
            
        self.data_loader.set_mode("training_model")
        self.generator.train()
        self.generator.to(self.device)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            print(f"Beginning epoch {self.current_epoch}")
            self.data_loader.shuffle_data()

            for idx, batch in enumerate(self.data_loader.get_batches()):
                # Move batch to device
                ct = batch.ct.to(self.device)
                masks = batch.structure_masks.to(self.device)
                dose = batch.dose.to(self.device).permute(0, 4, 1, 2, 3)
                
                # Forward pass
                self.gen_optimizer.zero_grad()
                output = self.generator(ct, masks)
                loss = self.criterion(output, dose)

                # Backward pass and optimize
                loss.backward()
                self.gen_optimizer.step()

                print(f"Model loss at epoch {self.current_epoch} batch {idx} is {loss.item():.3f}")

            self.manage_model_storage(save_frequency, keep_model_history)

    def _set_epoch_start(self) -> None:
        all_model_paths = get_paths(self.model_dir, extension="pt")  # Changed from h5 to pt
        for model_path in all_model_paths:
            *_, epoch_number = model_path.stem.split("epoch_")
            if epoch_number.isdigit():
                self.current_epoch = max(self.current_epoch, int(epoch_number))

    def initialize_networks(self) -> None:
        if self.current_epoch >= 1:
            self.generator = torch.load(self._get_generator_path(self.current_epoch),weights_only=False)
            self.gen_optimizer = self.gen_optimizer(self.generator.parameters())
        else:
            self.generator = self.define_generator()
            self.gen_optimizer = self.gen_optimizer(self.generator.parameters())

    def manage_model_storage(self, save_frequency: int = 1, keep_model_history: Optional[int] = None) -> None:
        effective_epoch_number = self.current_epoch + 1
        if 0 < np.mod(effective_epoch_number, save_frequency) and effective_epoch_number != self.last_epoch:
            Warning(f"Model at the end of epoch {self.current_epoch} was not saved because it is skipped when save frequency {save_frequency}.")
            return

        epoch_to_overwrite = effective_epoch_number - keep_model_history * (save_frequency or float("inf"))
        if epoch_to_overwrite >= 0:
            initial_model_path = self._get_generator_path(epoch_to_overwrite)
            torch.save(self.generator, initial_model_path)
            os.rename(initial_model_path, self._get_generator_path(effective_epoch_number))
        else:
            torch.save(self.generator, self._get_generator_path(effective_epoch_number))

    def _get_generator_path(self, epoch: Optional[int] = None) -> Path:
        epoch = epoch or self.current_epoch
        return self.model_dir / f"epoch_{epoch}.pt"  # Changed from h5 to pt

    def predict_dose(self, epoch: int = 1) -> None:
        """Predicts the dose for the given epoch number"""
        self.generator = torch.load(self._get_generator_path(epoch),weights_only=False)
        self.generator.eval()
        self.generator.to(self.device)
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.data_loader.set_mode("dose_prediction")

        print("Predicting dose with generator.")
        with torch.no_grad():
            for batch in self.data_loader.get_batches():
                ct = batch.ct.to(self.device)
                masks = batch.structure_masks.to(self.device)
                
                dose_pred = self.generator(ct, masks).permute(0, 2, 3, 4, 1).cpu().numpy()
                
                dose_pred = dose_pred * batch.possible_dose_mask.numpy()
                dose_pred = np.squeeze(dose_pred)
                
                dose_to_save = sparse_vector_function(dose_pred)
                dose_df = pd.DataFrame(
                    data=dose_to_save["data"].squeeze(),
                    index=dose_to_save["indices"].squeeze(),
                    columns=["data"]
                )
                (patient_id,) = batch.patient_list
                dose_df.to_csv(f"{self.prediction_dir}/{patient_id}.csv")
"""Registry-based model factory with torch and Riemannian adapters."""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.custom_s4d import SimpleS4D

LOGGER = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """Common interface for all training and inference backends."""

    model_name: str

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        head_only: bool = False,
    ) -> dict[str, float]:
        """Train the model and return summary metrics."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray, mc_dropout_passes: int = 1) -> np.ndarray:
        """Predict class probabilities for one or more windows."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load persisted weights from disk."""


class TorchModelAdapter(BaseModelAdapter):
    """Simple training wrapper around PyTorch-based EEG models."""

    def __init__(self, model_name: str, model: nn.Module) -> None:
        self.model_name = model_name
        self.model = model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        head_only: bool = False,
    ) -> dict[str, float]:
        from sklearn.model_selection import train_test_split

        self._configure_trainable_layers(head_only=head_only)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_inputs = torch.tensor(X_val, dtype=torch.float32, device=self._device)
        val_targets = torch.tensor(y_val, dtype=torch.long, device=self._device)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        best_state = None
        best_val_loss = float("inf")
        best_val_acc = 0.0
        stagnant_epochs = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self._device)
                batch_targets = batch_targets.to(self._device)
                optimizer.zero_grad()
                logits = self.model(batch_inputs)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_inputs)
                val_loss = float(criterion(val_logits, val_targets).item())
                val_predictions = torch.argmax(val_logits, dim=1)
                val_acc = float((val_predictions == val_targets).float().mean().item())

            LOGGER.info(
                "Epoch %s/%s train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch + 1,
                epochs,
                train_loss / max(len(loader), 1),
                val_loss,
                val_acc,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                if stagnant_epochs >= patience:
                    LOGGER.info("Early stopping triggered after %s epochs", epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"val_loss": best_val_loss, "val_acc": best_val_acc}

    def predict_proba(self, X: np.ndarray, mc_dropout_passes: int = 1) -> np.ndarray:
        inputs = torch.tensor(X, dtype=torch.float32, device=self._device)
        passes = max(mc_dropout_passes, 1)
        outputs: list[np.ndarray] = []
        for _ in range(passes):
            if passes > 1:
                self.model.train()
            else:
                self.model.eval()
            with torch.no_grad():
                logits = self.model(inputs)
                probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            outputs.append(probabilities)
        return np.mean(np.stack(outputs, axis=0), axis=0)

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self._device)
        self.model.load_state_dict(state)
        self.model.to(self._device)

    def _configure_trainable_layers(self, head_only: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = not head_only
        if not head_only:
            return

        classifier = getattr(self.model, "classifier", None)
        if isinstance(classifier, nn.Module):
            for parameter in classifier.parameters():
                parameter.requires_grad = True
            return

        last_trainable: nn.Module | None = None
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                last_trainable = module
        if last_trainable is not None:
            for parameter in last_trainable.parameters():
                parameter.requires_grad = True


class RiemannMDMAdapter(BaseModelAdapter):
    """pyRiemann MDM classifier with softmax-normalized distance scores."""

    def __init__(self) -> None:
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances

        self.model_name = "riemann-mdm"
        self._covariances = Covariances(estimator="oas")
        self._classifier = MDM(metric="riemann")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        head_only: bool = False,
    ) -> dict[str, float]:
        del epochs, batch_size, learning_rate, patience, head_only
        covs = self._covariances.fit_transform(X)
        
        # Add regularization to ensure positive definiteness, especially for short 
        # calibration data or high channel counts (e.g. 64) against small sample counts.
        # This prevents "Matrices must be positive definite" errors.
        n_channels = covs.shape[1]
        reg_amount = 1e-4 * np.trace(covs.mean(axis=0)) / n_channels
        covs += np.eye(n_channels) * reg_amount
        
        self._classifier.fit(covs, y)
        predictions = self._classifier.predict(covs)
        accuracy = float(np.mean(predictions == y))
        return {"val_loss": 0.0, "val_acc": accuracy}

    def predict_proba(self, X: np.ndarray, mc_dropout_passes: int = 1) -> np.ndarray:
        del mc_dropout_passes
        covs = self._covariances.transform(X)
        
        # Apply the same regularization at inference time
        n_channels = covs.shape[1]
        reg_amount = 1e-4 * np.trace(covs.mean(axis=0)) / n_channels
        covs += np.eye(n_channels) * reg_amount
        
        distances = self._classifier.transform(covs)
        return softmax(-distances, axis=1)

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "covariances": self._covariances,
                    "classifier": self._classifier,
                },
                handle,
            )

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._covariances = payload["covariances"]
        self._classifier = payload["classifier"]


class ModelFactory:
    """Model registry for all built-in motor imagery decoders."""

    @staticmethod
    def get(
        model_name: str,
        n_chans: int,
        sfreq: float,
        n_classes: int = 3,
        n_times: int | None = None,
    ) -> BaseModelAdapter:
        n_times = n_times or int(sfreq * 4.0)

        if model_name == "riemann-mdm":
            return RiemannMDMAdapter()
        if model_name == "s4d":
            return TorchModelAdapter(model_name, SimpleS4D(n_chans, n_times, n_classes))
        if model_name in {"eegnet", "deepconvnet", "shallowconvnet"}:
            return TorchModelAdapter(
                model_name,
                _build_braindecode_model(
                    model_name=model_name,
                    n_chans=n_chans,
                    n_times=n_times,
                    n_classes=n_classes,
                    sfreq=sfreq,
                ),
            )
        raise ValueError(
            "Unknown model '%s'. Available models: eegnet, deepconvnet, "
            "shallowconvnet, riemann-mdm, s4d" % model_name
        )

    @staticmethod
    def list_models() -> list[str]:
        return ["deepconvnet", "eegnet", "riemann-mdm", "s4d", "shallowconvnet"]


def _build_braindecode_model(
    *,
    model_name: str,
    n_chans: int,
    n_times: int,
    n_classes: int,
    sfreq: float,
) -> nn.Module:
    _ensure_moabb_dataset_aliases()
    from braindecode.models import Deep4Net, EEGNetv4, ShallowFBCSPNet

    kwargs = {
        "n_chans": n_chans,
        "n_outputs": n_classes,
        "n_times": n_times,
        "sfreq": sfreq,
        "input_window_seconds": n_times / sfreq,
        "final_conv_length": "auto",
    }
    constructors: dict[str, type[nn.Module]] = {
        "eegnet": EEGNetv4,
        "deepconvnet": Deep4Net,
        "shallowconvnet": ShallowFBCSPNet,
    }
    constructor = constructors[model_name]

    try:
        return constructor(**kwargs)
    except (TypeError, ValueError) as exc:
        LOGGER.debug(
            "Retrying braindecode model construction without input_window_seconds: %s",
            exc,
        )

    relaxed_kwargs = {
        "n_chans": n_chans,
        "n_outputs": n_classes,
        "n_times": n_times,
        "sfreq": sfreq,
        "final_conv_length": "auto",
    }
    try:
        return constructor(**relaxed_kwargs)
    except TypeError as exc:
        LOGGER.debug(
            "Retrying braindecode model construction without sfreq: %s",
            exc,
        )

    # Braindecode constructor signatures vary slightly across releases.
    reduced_kwargs = {
        "n_chans": n_chans,
        "n_outputs": n_classes,
        "n_times": n_times,
        "final_conv_length": "auto",
    }
    return constructor(**reduced_kwargs)


def _ensure_moabb_dataset_aliases() -> None:
    """Bridge MOABB dataset renames expected by some braindecode releases."""

    try:
        import moabb.datasets as moabb_datasets
    except Exception:  # noqa: BLE001
        return

    if hasattr(moabb_datasets, "BNCI2014001"):
        return

    renamed = getattr(moabb_datasets, "BNCI2014_001", None)
    if renamed is not None:
        moabb_datasets.BNCI2014001 = renamed

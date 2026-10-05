import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import os
import time
from models.models import JetEdgeConv

def build_graph(row, k=7):
    """row: (200, 4) array with columns (E, PX, PY, PZ) => PyG Data(x, edge_index)."""
    mask = row[:, 0] > 0                      # drop zero-padded particles
    particles = row[mask]
    E, PX, PY, PZ = particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3]

    pT  = np.sqrt(PX**2 + PY**2)
    p   = np.sqrt(PX**2 + PY**2 + PZ**2)
    eta = np.arctanh(np.clip(PZ / (p + 1e-8), -1 + 1e-7, 1 - 1e-7))
    phi = np.arctan2(PY, PX)

    pT_sum = pT.sum() + 1e-8
    E_sum  = E.sum() + 1e-8
    eta_jet = (pT * eta).sum() / pT_sum       # pT-weighted jet axis
    phi_jet = (pT * phi).sum() / pT_sum

    pT_rel    = pT / pT_sum
    delta_eta = eta - eta_jet
    delta_phi = (phi - phi_jet + np.pi) % (2 * np.pi) - np.pi
    E_rel     = E / E_sum

    x = np.stack([pT_rel, delta_eta, delta_phi, E_rel], axis=1)

    pos = np.stack([delta_eta, delta_phi], axis=1)
    k_actual = min(k, len(pos) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(pos)
    _, indices = nbrs.kneighbors(pos)
    src = np.repeat(np.arange(len(pos)), k_actual + 1)
    dst = indices.flatten()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)

class Inferencer:
    def __init__(self, model_path: str) -> None:

        self.model = JetEdgeConv()

        # Verify that the model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def predict(self, raw_jet: np.ndarray) -> dict:
        """
        predict the class of a single jet.
        """

        # verify that the input is a numpy array of shape (200, 4)
        try:
            self._validate_input(raw_jet)
        except TypeError:
            raw_jet = np.array(raw_jet, dtype=np.float32)
            self._validate_input(raw_jet)

        start = time.perf_counter()

        with torch.no_grad():
            data = build_graph(raw_jet)
            batch = torch.zeros(data.x.size(0), dtype=torch.long) # single graph, so batch is all zeros
            output = self.model(data.x, data.edge_index, batch)
            score = torch.softmax(output, dim=1)[0, 1].item()
            prediction = int(score > 0.5)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            'score': score,
            'prediction': prediction,
            'latency_ms': latency_ms
        }

    def predict_batch(self, raw_jets: list[np.ndarray]) -> dict: 
        """
        Predict the class of a batch of jets.
        """
        start = time.perf_counter()
        results = [self.predict(raw_jet) for raw_jet in raw_jets]
        elapsed = (time.perf_counter() - start)

        return {
            'results': results,
            'throughput': len(raw_jets) / elapsed if elapsed > 0 else 0.0,
            'total_ms': elapsed * 1000
        }


    def is_ready(self) -> bool:
        return self.model is not None and not self.model.training
        

    def _validate_input(self, raw_jet: np.ndarray):

        """
        Validate the input jet.
        """
        if not isinstance(raw_jet, np.ndarray):
            raise TypeError(f"Invalid input type: expected np.ndarray, got {type(raw_jet)}")
        if raw_jet.shape != (200, 4):
            raise ValueError(f"Invalid input shape: expected (200, 4), got {raw_jet.shape}")
        if not np.any(raw_jet[:, 0] > 0):
            raise ValueError("Input jet contains no non-zero particles.")


        
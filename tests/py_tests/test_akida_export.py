import pytest
import numpy as np
from pathlib import Path

# Make imports optional so the test runner can run on hosts without PyTorch or BrainChip SDK,
# though we skip if not available.
try:
    import torch
    from neucode.architectures import HybridControlSNN, PopulationControlSNN
    torch_available = True
except ImportError:
    torch_available = False

try:
    import tf_keras as keras
    import cnn2snn
    import akida
    akida_sdk_available = True
except ImportError:
    try:
        import keras
        import cnn2snn
        import akida
        akida_sdk_available = True
    except ImportError:
        akida_sdk_available = False

from neucode.exporters import AKD1000Exporter
from neucode.controllers import AkidaController, KerasController

@pytest.fixture
def dummy_scaler_file(tmp_path):
    scaler_path = tmp_path / "dummy_scaler.npz"
    # SNN scaler expects data_min, data_scale, target_scale, and optionally deriv_clip
    np.savez(
        scaler_path,
        data_min=np.zeros((5,), dtype=np.float32),
        data_scale=np.ones((5,), dtype=np.float32),
        target_scale=10.0,
        deriv_clip=np.array([5.0], dtype=np.float32)
    )
    return scaler_path

@pytest.mark.skipif(not (torch_available and akida_sdk_available), reason="Requires PyTorch and BrainChip SDK")
def test_akida_export_hybrid(tmp_path, dummy_scaler_file):
    # Instantiate HybridControlSNN
    model = HybridControlSNN(hidden_size=64)
    model.eval()

    exporter = AKD1000Exporter()
    output_path = tmp_path / "hybrid_model.fbz"

    # Export without QAT dataset
    exporter.export(model, str(output_path))

    # Verify fbz exists
    assert output_path.exists()

    # Load via AkidaController
    controller = AkidaController(model_path=str(output_path), scaler_path=str(dummy_scaler_file))
    pred = controller.predict(setpoint=1.0, measurement=0.5, error=0.5)
    assert isinstance(pred, float)

@pytest.mark.skipif(not (torch_available and akida_sdk_available), reason="Requires PyTorch and BrainChip SDK")
def test_akida_export_population_with_qat(tmp_path, dummy_scaler_file):
    # Instantiate PopulationControlSNN
    model = PopulationControlSNN(hidden_size=64, population_size=32)
    model.eval()

    exporter = AKD1000Exporter()
    output_path = tmp_path / "population_model.fbz"

    # Prepare dummy QAT dataset (10 samples)
    rng = np.random.default_rng(42)
    dataset_x = rng.random((10, 7)).astype(np.float32)
    dataset_y = rng.random((10, 1)).astype(np.float32)

    # Export with QAT dataset (runs _finetune)
    exporter.export(model, str(output_path), dataset_x=dataset_x, dataset_y=dataset_y, epochs=2)

    # Verify output files
    assert output_path.exists()
    assert (tmp_path / "akida_input_scaler.npz").exists()
    assert (tmp_path / "keras_model").exists()

    # Load via AkidaController
    controller = AkidaController(model_path=str(output_path), scaler_path=str(dummy_scaler_file))
    pred = controller.predict(setpoint=1.0, measurement=0.5, error=0.5)
    assert isinstance(pred, float)

    # Load via KerasController
    keras_controller = KerasController(model_path=str(tmp_path / "keras_model"), scaler_path=str(dummy_scaler_file))
    kpred = keras_controller.predict(setpoint=1.0, measurement=0.5, error=0.5)
    assert isinstance(kpred, float)

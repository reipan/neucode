import pytest
import numpy as np

# Make torch an optional import for testing
try:
    import torch
    from neucode.architectures import NoContextMLPArchitecture
    torch_available = True
except ImportError:
    torch_available = False

from neucode.controllers import PIDController, ANNController

def test_pid_controller_init_success():
    """
    Test successful initialization of PIDController with valid gains only.
    """
    pid = PIDController(kp=2.0, ki=1.0, kd=0.05)
    assert pid.params == {'kp': 2.0, 'ki': 1.0, 'kd': 0.05}

def test_pid_controller_invalid_gains_error():
    """
    Test that initializing PIDController with invalid gains raises an appropriate error.
    """
    with pytest.raises(TypeError, match="must be numeric"):
        PIDController(kp=1.0, ki='invalid_gain', kd=3.0)

@pytest.fixture
def dummy_ann_files(tmp_path):
    """
    Pytest fixture to create a dummy model and scaler file for testing ANNController.
    This fixture will be skipped if PyTorch is not installed.
    """
    if not torch_available:
        pytest.skip("PyTorch not installed, skipping ANN controller test.")

    model_path = tmp_path / "dummy_model.pth"
    scaler_path = tmp_path / "dummy_scaler.npz"

    # Create and save a dummy model state dictionary
    dummy_model = NoContextMLPArchitecture(input_size=5)
    torch.save(dummy_model.state_dict(), model_path)

    # Create and save dummy scaler parameters (mean and scale)
    dummy_mean = np.zeros((5,), dtype=np.float32)
    dummy_scale = np.ones((5,), dtype=np.float32)
    np.savez(scaler_path, mean=dummy_mean, scale=dummy_scale)

    return model_path, scaler_path

def test_ann_controller_init_and_predict(dummy_ann_files):
    """
    Tests that the ANNController can be initialized with valid files
    and that its predict method returns a float.
    """
    model_path, scaler_path = dummy_ann_files

    # Test successful initialization
    controller = ANNController(model_path=model_path, scaler_path=scaler_path)
    assert isinstance(controller, ANNController)

    # Test that predict returns a single float value
    prediction = controller.predict(setpoint=1.0, measurement=0.5, error=0.5)
    assert isinstance(prediction, float)
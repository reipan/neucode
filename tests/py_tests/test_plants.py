import pytest
from neucode.plants import FOPDTPlant

def test_plant_fpodt_init_success():
    """
    Test successful initialization of FOPDT plant with valid parameters.
    """
    plant = FOPDTPlant(K=2.0, tau=5.0, theta=1.0)
    assert plant.params == {'K': 2.0, 'tau': 5.0, 'theta': 1.0, 'friction': 0.0}

def test_plant_fpodt_invalid_params_error():
    """
    Test that initializing FOPDT plant with invalid parameters raises an appropriate error.
    """
    with pytest.raises(ValueError):
        FOPDTPlant(K='invalid', tau=5.0, theta=1.0)
    with pytest.raises(ValueError):
        FOPDTPlant(K=2.0, tau=-5.0, theta=1.0)
    with pytest.raises(ValueError):
        FOPDTPlant(K=2.0, tau=5.0, theta=-1.0)
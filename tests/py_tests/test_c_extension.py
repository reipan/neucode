"""
Test for C extension module import and instantiation.
"""
def test_c_extension_import_and_instantiation():
    """
    Tests that the core 'Simulation' object can be successfully instantiated
    (neucode_sim_create) and garbage collected (neucode_sim_destroy)
    without raising exceptions.
    """
    try:
        from neucode.simcore import Simulation
    except MemoryError as e:
        assert False, f"Failed to import the compiled C extension: {e}"

    sim_instance = Simulation()
    assert sim_instance is not None, "Simulation object should not be None after creation."
import os
import json
from pathlib import Path
from typing import Dict

class Experiment:
    """
    Artifact factory to standardize the storage and retrieval of experiment artifacts across the lifecycle of a project.

    Structure:
    - data/: Telemetry and Datasets (organized by tag)
    - training/: Models, Scalers, Stats and Training artifacts (organized by tag)
    - deployment/: Generated C Headers (organized by tag), Firmware deployment targets (organized by type)
    """
    def __init__(self, name: str, base_dir: str = None):
        self.name = name
        if base_dir is None:
            project_root = Path(__file__).parent.parent
            base_dir = project_root / "experiments"
        self.toolkit_root = Path(base_dir).resolve()
        self.experiment_root = self.toolkit_root / self.name

        # Lifecycle directories
        self.directories = {
            'data': self.experiment_root / 'data',
            'training': self.experiment_root / 'training',
            'deployment': self.experiment_root / 'deployment',
        }

        # Make sure all directories exist
        for path in self.directories.values():
            path.mkdir(parents=True, exist_ok=True)

        self.config_path = self.experiment_root / 'config.json'
        self.config = self._load_config()

    def _load_config(self):
        """
        Load experiment configuration from project's config.json if it exists, otherwise return default config.
        The config is used both for the simulation and the real-device experiments as a single source of truth.
        """
        if self.config_path.exists():
            import json
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "experiment_name": self.name,
                "created_at": str(os.path.getctime(self.experiment_root)) if self.experiment_root.exists() else None,
                "timestamp": None,
                "actuator_limits": {
                    "u_min": -1.0,
                    "u_max": 1.0,
                    "i_min": -0.5,
                    "i_max": 0.5,
                    "kaw": None,
                    "d_alpha": None
                } 
            }
        
    def save_config(self):
        """
        Save the current configuration to the project's config.json file.
        """
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    # Setter/Getter for config parameters
    def set_actuator_limits(self, limits: Dict[str, float]):
        """
        Update the actuator limits in the experiment configuration.

        :param limits: A dictionary containing any of the following keys: 'u_min', 'u_max', 'i_min', 'i_max', 'kaw', 'd_alpha'.
        """
        current_limits = self.config.get("actuator_limits", {})
        current_limits.update(limits)
        self.config["actuator_limits"] = current_limits
        self.save_config()

    @property
    def actuator_limits(self) -> Dict[str, float]:
        """
        Retrieve the actuator limits from the experiment configuration.

        :return: A dictionary containing the actuator limits.
        """
        return self.config.get("actuator_limits", {})

    def set_plant_params(self, params: dict):
        """
        Store plant identification parameters in the experiment configuration.

        :param params: A dictionary of plant parameters (e.g. Kv, tau, theta, friction).
        """
        self.config["plant"] = params
        self.save_config()

    @property
    def plant_params(self) -> dict:
        """
        Retrieve plant parameters from the experiment configuration.

        :return: A dictionary containing plant parameters.
        """
        return self.config.get("plant", {})

    def set_training_defaults(self, defaults: dict):
        """
        Store per-architecture training hyperparameter defaults.

        :param defaults: A dictionary keyed by architecture type (e.g. 'ann', 'snn')
            mapping to dicts of hyperparameters.
        """
        self.config["training_defaults"] = defaults
        self.save_config()

    @property
    def training_defaults(self) -> dict:
        """
        Retrieve per-architecture training defaults from the experiment configuration.

        :return: A dictionary keyed by architecture type (e.g. 'ann', 'snn').
        """
        return self.config.get("training_defaults", {})

    @property
    def data_dir(self) -> str:
        """
        Get the path to the data directory for this experiment.
        
        :return: Path to the data directory as a string.
        """
        return str(self.directories['data'])

    def get_telemetry_path(self, tag: str = "default") -> str:
        """
        Get the path to the telemetry CSV file for a given tag.

        :param tag: A string tag to differentiate telemetry files.
        :return: Path to the telemetry CSV file as a string.
        """
        return str(self.directories['data'] / f"{tag}_telemetry.csv")
    
    def get_telemetry_meta_path(self, tag: str = "default") -> str:
        """
        Get the path to the telemetry metadata JSON file for a given tag.

        :param tag: A string tag to differentiate telemetry metadata files.
        :return: Path to the telemetry metadata JSON file as a string.
        """
        return str(self.directories['data'] / f"{tag}_telemetry_meta.json")

    def get_dataset_path(self, tag: str = "default") -> str:
        """
        Get the path to the dataset CSV file for a given tag.

        :param tag: A string tag to differentiate dataset files.
        :return: Path to the dataset CSV file as a string.
        """
        return str(self.directories['data'] / f"{tag}_dataset.csv")
    
    def get_tmp_path(self, filename: str) -> str:
        """
        Get the path to a temporary file in the data directory.

        :param filename: The name of the temporary file.
        :return: Path to the temporary file as a string.
        """
        return str(self.directories['data'] / filename)
    
    # Training
    def _get_train_dir(self, tag: str) -> Path:
        """
        Get the directory path for training artifacts for a given tag. Creates the directory if it doesn't exist.

        :param tag: A string tag to differentiate training runs.
        :return: Path to the training artifacts directory as a Path object.
        """
        path = self.directories['training'] / tag
        path.mkdir(exist_ok=True)
        return path

    def get_model_path(self, tag: str) -> str:
        """
        Get the path to the model file for a given tag.

        :param tag: A string tag to differentiate model files.
        :return: Path to the model file as a string.
        """
        return str(self._get_train_dir(tag) / 'model.pth')
    
    def get_scaler_path(self, tag: str) -> str:
        """
        Get the path to the scaler file for a given tag.

        :param tag: A string tag to differentiate scaler files.
        :return: Path to the scaler file as a string.
        """
        return str(self._get_train_dir(tag) / 'scaler.npz')

    def get_stats_path(self, tag: str) -> str:
        """
        Get the path to the training stats JSON file for a given tag.

        :param tag: A string tag to differentiate stats files.
        :return: Path to the training stats JSON file as a string.
        """
        return str(self._get_train_dir(tag) / 'stats.json')

    # Deployment
    def get_model_data_header_path(self, tag: str) -> str:
        """
        Get the path to the generated C header file for a given tag.
        
        :param tag: A string tag to differentiate deployment files.
        :return: Path to the generated C header file as a string.
        """
        return str(self.directories['deployment'] / f"{tag}_model_data.h")
    
    def get_model_data_deployment_target_path(self, type: str) -> str:
        """
        Get the path to the firmware deployment target for a given controller type.
        
        :param type: The type of controller (e.g., "snn", "ann") to determine the deployment target.
        :return: Path to the firmware deployment target as a string.
        """
        project_root = Path(__file__).parent.parent
        if type == "snn":
            return str(project_root / "firmware/controller/snn/model_data.h")
        elif type == "ann":
            return str(project_root / "firmware/controller/ann/model_data.h")
        else:
            raise ValueError(f"Unknown deployment target type: {type}")

    # Controller factory
    @staticmethod
    def _detect_controller_type(model_path: str) -> str:
        """
        Infer controller/architecture type from saved state dict keys.
        """
        from neucode._torch_optional import torch
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        keys = set(state_dict.keys())
        has_context = any(k.startswith('fc_context.') for k in keys)
        has_spike_input = any(k.startswith('fc_input_spikes.') for k in keys)
        if has_context or has_spike_input:
            return 'hybrid'
        if any(k.startswith('output_population.') for k in keys):
            if any(k.startswith('fc_input.') for k in keys):
                return 'spike'
            return 'population'
        if any(k.startswith('encoder.') or k.startswith('fc_input.') for k in keys):
            return 'spike'
        return 'ann'

    def load_controller(self, tag: str, controller_type: str = None,
                        dt: float = None, **kwargs):
        """
        Load a trained controller from experiment artifacts.

        Auto-detects controller type from the model state dict if
        controller_type is not specified.

        :param tag: Training tag identifying the model within this experiment.
        :param controller_type: One of 'ann', 'hybrid', 'population', 'spike'.
            If None, detected automatically from saved weights.
        :param dt: Control loop timestep in seconds.  Falls back to
            ``config['dt']`` if not provided.
        :param kwargs: Extra keyword arguments forwarded to the controller
            constructor (e.g. filter_alpha, record_state).
        :returns: Configured controller instance, or None if the model file
            does not exist.
        """
        model_path = self.get_model_path(tag)
        scaler_path = self.get_scaler_path(tag)

        if not os.path.exists(model_path):
            return None

        if controller_type is None:
            controller_type = self._detect_controller_type(model_path)

        if dt is None:
            dt = self.config.get('dt', 0.01)

        actuator_limits = kwargs.pop('actuator_limits', None)
        if actuator_limits is None:
            actuator_limits = self.actuator_limits or None

        if controller_type == 'ann':
            from neucode.controllers import ANNController
            return ANNController(
                model_path=model_path,
                scaler_path=scaler_path,
                dt=dt,
                actuator_limits=actuator_limits,
                **kwargs,
            )

        from neucode.controllers import SNNController
        architecture = controller_type if controller_type != 'snn' else 'hybrid'
        return SNNController(
            model_path=model_path,
            scaler_path=scaler_path,
            dt=dt,
            actuator_limits=actuator_limits,
            architecture=architecture,
            **kwargs,
        )
"""Akida exporter for the NeuCoDe toolkit."""

from pathlib import Path
from .base import BaseExporter
from ..ui import get_progress_bar


class AKD1000Exporter(BaseExporter):
    """
    Exporter for SNN controllers -> BrainChip AKD1000 (.fbz).

    Named after the hardware target because the AKD1000 fully determines all
    export constraints: SNN-only execution, 4-bit weight quantization, and the
    BrainChip .fbz binary format. Contrast with ANNExporter/SNNExporter which
    are named by model type because their C-header output is MCU-agnostic.

    Call export() to run the full pipeline: weight fusion, float fine-tuning,
    quantization, and cnn2snn conversion.

    The SNN model must implement to_keras_weights() returning a dict with
    'W_fused', 'W_output', 'B_output', 'out_features', and 'hidden_size'.
    HybridControlSNN and PopulationControlSNN both implement this protocol.
    To export a custom architecture, implement to_keras_weights() on your model
    or subclass AKD1000Exporter and override _fuse_weights().

    All heavy dependencies (tensorflow, cnn2snn, akida) are imported lazily
    so NeuCoDe remains importable on machines without the BrainChip SDK.
    """

    @staticmethod
    def _apply_keras_compat():
        """
        Patch missing symbols in old keras.utils.generic_utils for cnn2snn 2.2.2.
        """
        try:
            import keras.utils.generic_utils as _gu
            _missing = [
                'serialize_keras_object', 'deserialize_keras_object',
                'get_registered_object', 'get_registered_name',
            ]
            if all(hasattr(_gu, fn) for fn in _missing):
                return
            for _mod_path in [
                'keras.saving.legacy.serialization',
                'keras.src.saving.legacy.serialization',
                'keras.saving.serialization_lib',
            ]:
                try:
                    import importlib
                    _mod = importlib.import_module(_mod_path)
                    for _fn in _missing:
                        if not hasattr(_gu, _fn) and hasattr(_mod, _fn):
                            setattr(_gu, _fn, getattr(_mod, _fn))
                except ImportError:
                    continue
            try:
                from keras.saving.object_registration import (
                    get_registered_object, get_registered_name
                )
                if not hasattr(_gu, 'get_registered_object'):
                    _gu.get_registered_object = get_registered_object
                if not hasattr(_gu, 'get_registered_name'):
                    _gu.get_registered_name = get_registered_name
            except ImportError:
                pass
        except ImportError:
            print("[info] cnn2snn keras compat not available -- "
                  "running outside akida-export Docker container, "
                  ".fbz output will not be usable on AKD1000 hardware")

    @staticmethod
    def _set_seeds(seed: int):
        """
        Set all random seeds for reproducible fine-tuning.
        """
        import os
        import random
        import numpy as np
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except Exception:
            pass
        try:
            import tf_keras as _k
            _k.utils.set_random_seed(seed)
        except Exception:
            try:
                import keras as _k
                _k.utils.set_random_seed(seed)
            except Exception:
                pass

    def _fuse_weights(self, model):
        """
        Build a fused Keras model from the SNN weight matrices.

        The model must implement to_keras_weights() returning a dict with:
        W_fused (7, hidden), W_output, B_output, out_features, hidden_size.

        :param model: Trained SNN instance with to_keras_weights().
        :returns: Float Keras Sequential model with weights transferred.
        :raises TypeError: If model does not implement to_keras_weights().
        """
        if not hasattr(model, 'to_keras_weights'):
            raise TypeError(
                f"{type(model).__name__} is not Akida-exportable. "
                "Implement to_keras_weights() returning "
                "{'W_fused', 'W_output', 'B_output', 'out_features', 'hidden_size'}."
            )

        try:
            import tf_keras as keras
        except ImportError:
            import keras

        weights      = model.to_keras_weights()
        W_fused      = weights['W_fused']
        hidden_size  = weights['hidden_size']
        W_output     = weights['W_output']
        B_output     = weights['B_output']
        out_features = weights['out_features']

        keras_model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(1, 1, 7), name="fused_input"),
            keras.layers.Conv2D(hidden_size, (1, 1), use_bias=False, padding='same', name="hidden_layer_1"),
            keras.layers.ReLU(max_value=1.0, name="hidden_relu_1"),
            keras.layers.Conv2D(out_features, (1, 1), use_bias=(B_output is not None), padding='same', name="output_layer"),
        ])

        # Conv2D weight shape: (kernel_h, kernel_w, in_channels, out_channels)
        keras_model.get_layer("hidden_layer_1").set_weights(
            [W_fused.reshape(1, 1, W_fused.shape[0], W_fused.shape[1])]
        )
        output_W = W_output.T.reshape(1, 1, W_output.shape[1], W_output.shape[0])
        if B_output is not None:
            keras_model.get_layer("output_layer").set_weights([output_W, B_output])
        else:
            keras_model.get_layer("output_layer").set_weights([output_W])

        return keras_model

    def _quantize(self, keras_model):
        """
        Quantize a Keras model with cnn2snn.quantize().

        Uses 8-bit weights and 4-bit activations.

        :param keras_model: Fine-tuned float Keras model from _finetune().
        :returns: Quantized Keras model.
        """
        import cnn2snn
        # AKD1000 requires 4-bit weights for hardware execution.
        return cnn2snn.quantize(
            keras_model,
            weight_quantization=4,
            activ_quantization=4,
            input_weight_quantization=4,
        )

    def _finetune(self, keras_model, dataset_x, dataset_y, epochs=10):
        """
        Fine-tune the float Keras model on the replacement dataset.

        For population models (out_features > 1), a Lambda layer computes the
        mean over the population axis so the loss is compared against the scalar target.

        :param keras_model: Float Keras model from _fuse_weights().
        :param dataset_x: Normalised input array, shape (N, 7).
        :param dataset_y: Target scalar array, shape (N, 1).
        :param epochs: Number of fine-tuning epochs.
        :returns: Fine-tuned float Keras model.
        """
        try:
            import tf_keras as keras
        except ImportError:
            import keras
        try:
            import tensorflow as tf
        except ImportError:
            tf = None
        import numpy as np

        out_features = keras_model.get_layer("output_layer").get_weights()[0].shape[3]
        x_train = dataset_x.reshape(-1, 1, 1, 7).astype(np.float32)
        y_train = dataset_y.reshape(-1, 1, 1, 1).astype(np.float32)

        inputs = keras.layers.Input(shape=(1, 1, 7))
        x = keras_model(inputs)
        if out_features > 1:
            if tf is not None:
                outputs = keras.layers.Lambda(lambda val: tf.reduce_mean(val, axis=-1, keepdims=True))(x)
            else:
                outputs = keras.layers.Lambda(lambda val: keras.backend.mean(val, axis=-1, keepdims=True))(x)
        else:
            outputs = x

        n_samples = len(x_train)
        batch_size = 32
        steps_per_epoch = max(1, int(n_samples * 0.9) // batch_size)
        total_steps = steps_per_epoch * epochs

        class _RichProgressCallback(keras.callbacks.Callback):
            def __init__(self, pbar):
                super().__init__()
                self._pbar = pbar

            def on_batch_end(self, batch, logs=None):
                self._pbar.update(1)

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                status = (f"Epoch {epoch+1}/{epochs} | "
                          f"loss: {logs.get('loss', 0):.6f} | "
                          f"val_loss: {logs.get('val_loss', 0):.6f}")
                self._pbar.update(0, status=status)

        training_model = keras.models.Model(inputs=inputs, outputs=outputs)
        training_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
        with get_progress_bar("QAT fine-tuning", total=total_steps) as pbar:
            training_model.fit(
                x=x_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=3,
                        min_lr=1e-6, verbose=0),
                    _RichProgressCallback(pbar),
                ],
                verbose=0,
            )

        return keras_model

    def _calibrate_output_bias(self, fbz_path, keras_model, dataset_x_norm,
                                output_dir, n_samples=2000):
        """
        Measure the systematic output bias between the Akida and Keras models.

        4-bit quantization introduces a small constant offset. On integrating
        plants this accumulates into large steady-state error, so we measure it
        here and save it for AkidaController to compensate.

        :param fbz_path: Path to the exported .fbz model.
        :param keras_model: The float Keras model (same weights, before quantization).
        :param dataset_x_norm: Normalised QAT input array, shape (N, 7), values in [0, 1].
        :param output_dir: Directory to save akida_output_bias.npz.
        :param n_samples: Number of calibration samples (default 2000).
        """
        import akida
        import numpy as np

        print("* Calibrating output bias (Akida vs Keras)...")
        akida_mdl = akida.Model(str(fbz_path))
        n = min(n_samples, len(dataset_x_norm))
        x_sub = dataset_x_norm[:n]

        keras_input = x_sub.reshape(-1, 1, 1, 7).astype(np.float32)
        keras_preds = keras_model.predict(keras_input, verbose=0).mean(axis=-1).ravel()

        x_uint8 = np.clip(x_sub * 15, 0, 15).astype(np.uint8).reshape(-1, 1, 1, 7)
        akida_preds = np.array([
            akida_mdl.predict(x_uint8[i:i+1]).ravel().mean()
            for i in range(n)
        ])

        bias = float(np.mean(akida_preds - keras_preds))
        np.savez(str(Path(output_dir) / "akida_output_bias.npz"), bias=np.float32(bias))
        print(f"  Measured bias: {bias:+.6f} (over {n} samples)")

    def _verify(self, fbz_path, dataset_x, output_dir):
        """
        Reload the .fbz and run one inference to confirm the model is valid.
        """
        import akida
        import numpy as np

        print("* Verifying exported model...")
        reloaded   = akida.Model(str(fbz_path))
        test_input = dataset_x[:1] if dataset_x is not None else np.zeros((1, 7), dtype=np.float32)

        scaler_path = Path(output_dir) / "akida_input_scaler.npz"
        if scaler_path.exists():
            with np.load(str(scaler_path)) as sc:
                norm_input = np.clip((test_input - sc['feat_min']) / sc['feat_range'], 0.0, 1.0)
        else:
            norm_input = np.clip(test_input, 0.0, 1.0)

        test_uint8 = np.clip(norm_input * 15, 0, 15).astype(np.uint8).reshape(1, 1, 1, 7)
        prediction = reloaded.predict(test_uint8)
        print(f"  Akida prediction: {prediction.ravel().mean():.6f}")

    def _generate_calibration_data(self, harness, total_time=15.0, seed=42):
        """
        Generate QAT calibration data by running a SimulationHarness over a
        MultiStepSetpoint sequence and collecting the controller's recorded inputs.

        The simulation timestep is read from harness.controller.dt. Using any
        other dt would corrupt the calibration distribution.
        The record_inputs flag is enabled automatically if the controller supports it.

        :param harness: SimulationHarness with a MultiStepSetpoint as its setpoint.
        :param total_time: Duration of each episode in seconds (default 15.0).
        :param seed: RNG seed for shuffling the collected data (default 42).
        :returns: Tuple (x, y) of shuffled calibration arrays.
        :raises TypeError: If harness.setpoint is not a MultiStepSetpoint.
        :raises AttributeError: If the controller does not implement get_recorded_dataset().
        """
        from neucode.signals import MultiStepSetpoint
        import numpy as np

        if not isinstance(harness.setpoint, MultiStepSetpoint):
            raise TypeError(
                f"calibration_harness.setpoint must be a MultiStepSetpoint, "
                f"got {type(harness.setpoint).__name__}."
            )

        controller = harness.controller
        effective_dt = getattr(controller, 'dt', 0.01)
        print(f"* Calibration dt={effective_dt} (from controller)")
        if not hasattr(controller, 'get_recorded_dataset'):
            raise AttributeError(
                f"{type(controller).__name__} does not support input recording. "
                "Use an SNNController."
            )
        if hasattr(controller, 'enable_recording'):
            controller.enable_recording()

        setpoints = list(harness.setpoint)
        x_parts, y_parts = [], []
        with get_progress_bar("QAT calibration", total=len(setpoints)) as pbar:
            for step_setpoint in setpoints:
                controller.reset()
                harness.set_setpoint(step_setpoint)
                harness.run(dt=effective_dt, total_time=total_time)
                X, y = controller.get_recorded_dataset()
                x_parts.append(X)
                y_parts.append(y)
                pbar.update(status=f"sp={step_setpoint.config['v']:.1f}")

        x = np.concatenate(x_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x))
        print(f"  Generated {len(x)} calibration samples across {len(x_parts)} episodes.")
        return x[idx], y[idx]

    def export(
        self,
        model,
        output_path,
        dataset_x=None,
        dataset_y=None,
        dataset_path=None,
        dataset_fn=None,
        calibration_harness=None,
        calibration_total_time=15.0,
        epochs=10,
        seed=42,
        skip_finetune=False,
        verify=True,
    ):
        """
        Run the full export pipeline and save the Akida model as a .fbz file.

        Steps:
            1. _fuse_weights(model)      -> float Keras model
            2. Resolve dataset           -> from arrays, path cache, calibration_harness, or dataset_fn
            3. _finetune(...)            -> fine-tuned float Keras model (skipped if no data)
            4. _quantize(keras_model)    -> quantized Keras model
            5. cnn2snn.convert(...)      -> akida.Model
            6. akida_model.save(...)     -> .fbz
            7. _verify(...)              -> reload and run one inference (optional)

        Dataset resolution order (first match wins):
            1. dataset_x / dataset_y provided directly.
            2. dataset_path exists on disk - loaded from the .npz.
            3. calibration_harness provided - runs multi-episode simulation via MultiStepSetpoint;
               result saved to dataset_path if given.
            4. dataset_fn() called to generate data; result saved to dataset_path if given.
            5. No data - fine-tuning skipped with a warning.

        The fine-tuned float model is saved alongside the .fbz as keras_model/
        for use with KerasController.

        :param model: Trained SNN instance implementing to_keras_weights().
        :param output_path: Destination path for the .fbz file.
        :param dataset_x: Input array for fine-tuning, shape (N, 7). Optional.
        :param dataset_y: Target array for fine-tuning, shape (N, 1). Optional.
        :param dataset_path: Path to a .npz file to load or save the dataset cache.
        :param dataset_fn: Callable returning (x, y) arrays for on-the-fly generation.
        :param calibration_harness: SimulationHarness with a MultiStepSetpoint setpoint.
            Runs one episode per step value and collects controller-recorded inputs.
            dt is read from the controller - using a different dt would corrupt calibration.
        :param calibration_total_time: Duration of each calibration episode (default 15.0).
        :param epochs: Number of fine-tuning epochs (default 10).
        :param seed: Random seed for reproducible fine-tuning (default 42).
        :param skip_finetune: Skip fine-tuning entirely (default False).
        :param verify: Reload the .fbz and run one inference after export (default True).
        """
        import numpy as np

        # Patch keras.utils.generic_utils BEFORE importing cnn2snn
        # cnn2snn.__init__ imports serialize_keras_object immediately on import.
        self._apply_keras_compat()
        self._set_seeds(seed)

        import cnn2snn

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        keras_model = self._fuse_weights(model)

        if not skip_finetune and (dataset_x is None or dataset_y is None):
            if dataset_path is not None and Path(dataset_path).exists():
                print(f"* Loading QAT dataset from {dataset_path}")
                data = np.load(str(dataset_path))
                dataset_x, dataset_y = data['x_qat'], data['y_qat']
                print(f"  {len(dataset_x)} samples loaded.")
            elif calibration_harness is not None:
                print("* Generating QAT dataset via calibration_harness...")
                dataset_x, dataset_y = self._generate_calibration_data(
                    calibration_harness, total_time=calibration_total_time, seed=seed
                )
                if dataset_path is not None:
                    np.savez(str(dataset_path), x_qat=dataset_x, y_qat=dataset_y)
                    print(f"  Saved to {dataset_path}")
            elif dataset_fn is not None:
                print("* Generating QAT dataset via dataset_fn...")
                dataset_x, dataset_y = dataset_fn()
                if dataset_path is not None:
                    np.savez(str(dataset_path), x_qat=dataset_x, y_qat=dataset_y)
                    print(f"  {len(dataset_x)} samples saved to {dataset_path}")

        dataset_x_norm = None
        if skip_finetune:
            print("* Skipping fine-tuning (skip_finetune=True).")
        elif dataset_x is not None and dataset_y is not None:
            feat_min       = dataset_x.min(axis=0)
            feat_max       = dataset_x.max(axis=0)
            feat_range     = np.maximum(feat_max - feat_min, 1e-6)
            feat_min_pad   = feat_min   - 0.05 * feat_range
            feat_range_pad = feat_range * 1.10
            np.savez(str(output_dir / "akida_input_scaler.npz"),
                     feat_min=feat_min_pad, feat_range=feat_range_pad)

            dataset_x_norm = np.clip((dataset_x - feat_min_pad) / feat_range_pad, 0.0, 1.0)
            keras_model = self._finetune(keras_model, dataset_x_norm, dataset_y, epochs=epochs)
            keras_model.save(str(output_dir / "keras_model"))
        else:
            print("* No dataset available - skipping fine-tuning.")

        quantized_model = self._quantize(keras_model)

        import inspect
        sig = inspect.signature(cnn2snn.convert)
        if 'input_is_image' in sig.parameters:
            akida_model = cnn2snn.convert(quantized_model, input_scaling=(15, 0), input_is_image=False)
        else:
            akida_model = cnn2snn.convert(quantized_model, input_scaling=(15, 0))

        # Map to AKD1000 hardware architecture before saving.
        # akida.AKD1000() is a virtual device
        # hw_only=True fails fast if the model has any SW/ sequences,
        print("* Mapping to AKD1000 hardware...")
        import akida as _akida
        akida_model.map(device=_akida.AKD1000(), hw_only=True)
        print("  Hardware mapping successful - sequences:", [s.name for s in akida_model.sequences])

        out_path = Path(output_path).with_suffix('.fbz')
        akida_model.save(str(out_path))
        print(f"* Exported Akida model to {out_path}")

        if dataset_x is not None and dataset_x_norm is not None:
            self._calibrate_output_bias(
                out_path, keras_model, dataset_x_norm, output_dir)

        if verify:
            self._verify(out_path, dataset_x, output_dir)
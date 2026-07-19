"""
PlatformIO pre-build script: copy *_default.h stubs when generated headers are missing.

This mirrors the old CubeMX Makefile guards:
    @test -f model_data.h || cp model_data_default.h model_data.h

Allows building the firmware from a clean clone without running the ML training
pipeline first. The controller will load a zeroed-weight dummy model and produce
no output - safe for testing the build and basic communication, but the motor
will not move meaningfully until real models are exported via:
"""
import os
import shutil

Import("env")


def copy_if_missing(src: str, dst: str) -> None:
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"[pre_build] No exported model found - copied stub: {os.path.basename(src)} -> {dst}")
    else:
        print(f"[pre_build] Found: {dst}")


project_dir = env.subst("$PROJECT_DIR")
ctrl = os.path.join(project_dir, "..", "..", "controller")

# ANN: model_data.h
copy_if_missing(
    os.path.join(ctrl, "ann", "model_data_default.h"),
    os.path.join(ctrl, "ann", "model_data.h"),
)

# SNN: model_data.h and model_config.h
copy_if_missing(
    os.path.join(ctrl, "snn", "model_data_default.h"),
    os.path.join(ctrl, "snn", "model_data.h"),
)
copy_if_missing(
    os.path.join(ctrl, "snn", "model_config_default.h"),
    os.path.join(ctrl, "snn", "model_config.h"),
)

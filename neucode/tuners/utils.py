"""
Utility helpers for the NeuCoDe tuner sub-package.
"""
import warnings
import gymnasium as gym

try:
    from stable_baselines3.common.env_checker import check_env
    rl_libs_available = True
except ImportError:
    rl_libs_available = False
    def check_env(env):
        """
        No-op replacement for stable_baselines3 check_env when the library is unavailable.
        """
        print("Warning: stable_baselines3 not found. Skipping env check.")

def safe_check_env(env, verbose: int = 0):
    """
    Run SB3's check_env() while suppressing the normalized Box action-space warning.

    The warning is suppressed intentionally: experiments show that passing the
    unnormalized gains space directly to TD3/SAC works better than normalizing.
    PPO uses a RescaleAction wrapper instead and never triggers the warning.

    :param env: A Gymnasium environment instance to validate.
    :param verbose: Verbosity level; set >= 2 to re-enable the suppressed warning.
    """
    if not rl_libs_available:
        return

    with warnings.catch_warnings():
        if verbose < 2:
            # Intentionally ignored: we pass an unnormalized action space to
            # TD3/SAC on purpose, which suits this control task.
            warnings.filterwarnings(
                "ignore",
                message="We recommend you to use a symmetric and normalized Box action space",
                category=UserWarning,
            )
        check_env(env)
"""
Utility function to print device memory-usage
statistics (i.e., GPU VRAM) when using JAX.

Usually used during testing and profiling.
"""

import jax

__all__ = ["print_memory_stats"]


def print_memory_stats(label: str = "",
                       ) -> None:
    """
    Util function to print memory stats.
    """
    # Use JAX to get devices
    for device in jax.local_devices():
        print(f"Memory stats ({label}) {device}:")

        # Memory stats
        stats = device.memory_stats()
        if stats is None:
            print("  Not available...")
            continue

        # Peak, live and limit in GB
        peak = stats['peak_bytes_in_use'] / 1024**3
        live = stats['bytes_in_use'] / 1024**3
        lim = stats['bytes_limit'] / 1024**3

        print(f"  Live:  {live:.2f}/{lim:.2f} GB ({100*live/lim:.2f}%)")
        print(f"  Peak:  {peak:.2f}/{lim:.2f} GB ({100*peak/lim:.2f}%)")

"""Initialize shared animation helpers without requiring optional notebook cells.

The rendering functions intentionally share the caller's notebook namespace:
frame loops change motion/style controls and the callbacks must see those changes.
This initializer never renders or writes an output artifact.
"""
from pathlib import Path


def prepare_playback(namespace):
    """Refresh playback without resetting the notebook's current controls."""
    path = Path(__file__).with_name("_animation_playback.py")
    exec(compile(path.read_text(), str(path), "exec"), namespace)


def prepare_animation(namespace):
    required = ['base_peaks', 'variant_peaks', 'variant_overlay_peaks',
                'base_wave_field', 'variant_overlay_wave_field', 'MOTION_MODE',
                'OBSERVED_SPEED_REGULARIZATION', 'INLET_FEATURE_SECONDS', 'time_seconds',
                'REPO', 'SCALE_BAR_KM', 'DOT_SIZE', 'np', 'pd']
    missing = [key for key in required if key not in namespace]
    if missing:
        raise RuntimeError('Run the setup, data-loading and timing-calculation cells first. '
                           'Missing: ' + ', '.join(missing))
    prepare_playback(namespace)
    # Keep an overlay definition the user has edited in the notebook.
    overlay_override = namespace.get('draw_overlay')
    path = Path(__file__).with_name('_animation_notebook_helpers.py')
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    if overlay_override is not None:
        namespace['draw_overlay'] = overlay_override

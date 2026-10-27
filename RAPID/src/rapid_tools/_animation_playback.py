"""Playback helper executed in the notebook namespace to read live controls."""

def movie_playback(motion_mode, frame_count=None):
    """Apply view controls at render time; keep the original response clock."""
    window_hours = (ANIMATION_END_SECONDS - ANIMATION_START_SECONDS) / 3600
    start = 0.0 if MOVIE_START_HOUR is None else float(MOVIE_START_HOUR)
    end = window_hours if MOVIE_END_HOUR is None else float(MOVIE_END_HOUR)
    if not (0 <= start < end <= window_hours):
        raise ValueError(f'Movie bounds must satisfy 0 <= start < end <= {window_hours:.2f} hours; use None for full bounds.')
    count = frame_count if frame_count is not None else (HYBRID_N_FRAMES if motion_mode == 'hybrid_midpoint_timing' else N_FRAMES)
    if int(count) != count or count < 2:
        raise ValueError('At least two integer frames are required.')
    if not np.isfinite(PLAYBACK_SECONDS) or PLAYBACK_SECONDS <= 0:
        raise ValueError('PLAYBACK_SECONDS must be finite and positive.')
    # GIF stores 10 ms ticks, but viewers may replace 10 ms delays with 100 ms.
    # Keep every frame >=20 ms, reducing frame count for short playback.
    total_ticks = int(round(PLAYBACK_SECONDS * 100))
    if total_ticks < 4:
        raise ValueError('GIF playback must be at least 0.04 seconds.')
    count = min(int(count), total_ticks // 2)
    # Distribute whole ticks across frames so their sum matches the requested
    # duration (to the nearest 10 ms), instead of rounding every frame alike.
    tick_edges = np.arange(count + 1, dtype=np.int64) * total_ticks // count
    delays_ms = (np.diff(tick_edges) * 10).tolist()
    times = np.linspace(ANIMATION_START_SECONDS + start * 3600,
                        ANIMATION_START_SECONDS + end * 3600, count)
    print(f'Movie: window +{start:.2f} to +{end:.2f} h; '
          f'{count} frames, {sum(delays_ms) / 1000:.2f} s playback '
          f'(frame delays {min(delays_ms)}–{max(delays_ms)} ms)')
    return times, delays_ms

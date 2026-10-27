"""Drawing definitions evaluated by animation_notebook.prepare_animation in a notebook namespace."""
from shapely.geometry import LineString
from shapely.ops import linemerge, substring

def as_single_line(geometry):
    """Return a continuous line where possible; retain the longest part otherwise."""
    merged = linemerge(geometry) if geometry.geom_type == 'MultiLineString' else geometry
    if merged.geom_type == 'LineString':
        return merged
    return max(merged.geoms, key=lambda part: part.length)

def half_path(line, *, from_midpoint, endpoint):
    """Extract a curved half-link path, preserving all centreline vertices."""
    midpoint_distance = 0.5 * line.length
    endpoint_distance = float(endpoint) * line.length
    if from_midpoint:
        return substring(line, midpoint_distance, endpoint_distance)
    return substring(line, endpoint_distance, midpoint_distance)

def path_between_midpoints(source_geometry, target_geometry):
    """Join two link midpoints through their nearest endpoints.

    The graph topology selects valid downstream links; nearest endpoints only chooses
    the drawing direction along the stored centreline coordinates.
    """
    source, target = as_single_line(source_geometry), as_single_line(target_geometry)
    source_ends = [source.interpolate(0, normalized=True), source.interpolate(1, normalized=True)]
    target_ends = [target.interpolate(0, normalized=True), target.interpolate(1, normalized=True)]
    source_end, target_end = min(
        ((i, j) for i in range(2) for j in range(2)),
        key=lambda pair: source_ends[pair[0]].distance(target_ends[pair[1]]),
    )
    first = half_path(source, from_midpoint=True, endpoint=source_end)
    second = half_path(target, from_midpoint=False, endpoint=target_end)
    return LineString(list(first.coords) + list(second.coords)[1:])

def midpoint_timing_segments(peaks):
    """One mover reaches each downstream midpoint exactly at its peak time."""
    rows, downstream_nodes = [], set(peaks.id_ds_node.dropna().astype(int))
    for source in peaks.itertuples(index=False):
        downstream = peaks.loc[peaks.id_us_node.eq(source.id_ds_node)]
        for target_row in downstream.itertuples(index=False):
            if target_row.timing_time_seconds <= source.timing_time_seconds:
                continue  # no backwards/instantaneous visual path
            rows.append({
                'start_seconds': source.timing_time_seconds,
                'end_seconds': target_row.timing_time_seconds,
                'geometry': path_between_midpoints(source.geometry, target_row.geometry),
                'start_peak_width_seconds': source.peak_width_seconds,
                'peak_width_seconds': target_row.peak_width_seconds,
                'start_link_id': source.parent_link_id,
                'end_link_id': target_row.parent_link_id,
            })
    # The peak enters at the measured forcing peak, not at simulation/spinup start.
    for inlet in peaks.loc[~peaks.id_us_node.isin(downstream_nodes)].itertuples(index=False):
        if inlet.timing_time_seconds <= INLET_FEATURE_SECONDS:
            continue
        line = as_single_line(inlet.geometry)
        rows.append({
            'start_seconds': INLET_FEATURE_SECONDS,
            'end_seconds': inlet.timing_time_seconds,
            'geometry': half_path(line, from_midpoint=False, endpoint=0),
            'start_peak_width_seconds': inlet.peak_width_seconds,
            'peak_width_seconds': inlet.peak_width_seconds,
            'start_link_id': np.nan,
            'end_link_id': inlet.parent_link_id,
        })
    return pd.DataFrame(rows)

def hybrid_midpoint_segments(peaks):
    """Keep peak-time waypoints, while matching speed across shared midpoints."""
    movers = midpoint_timing_segments(peaks).copy()
    movers['geometry_length'] = movers.geometry.map(lambda geometry: geometry.length)
    movers['mean_speed'] = movers.geometry_length / (movers.end_seconds - movers.start_seconds)
    # Each link midpoint is a waypoint. Its harmonic-mean incident speed avoids
    # a sudden speed jump while preserving the exact waypoint time and location.
    waypoint_speeds = {}
    for link_id in peaks.parent_link_id:
        incident = pd.concat([
            movers.loc[movers.end_link_id.eq(link_id), 'mean_speed'],
            movers.loc[movers.start_link_id.eq(link_id), 'mean_speed'],
        ]).to_numpy(dtype=float)
        incident = incident[np.isfinite(incident) & (incident > 0)]
        if incident.size:
            waypoint_speeds[link_id] = incident.size / np.sum(1.0 / incident)
    movers['start_speed'] = [waypoint_speeds.get(link_id, speed) for link_id, speed in zip(movers.start_link_id, movers.mean_speed)]
    movers['end_speed'] = [waypoint_speeds.get(link_id, speed) for link_id, speed in zip(movers.end_link_id, movers.mean_speed)]
    duration = movers.end_seconds - movers.start_seconds
    # Slopes are d(distance fraction)/d(time fraction). Clipping keeps the
    # cubic interpolation monotonic even for strongly different link speeds.
    movers['start_slope'] = np.clip(movers.start_speed * duration / movers.geometry_length, 0.05, 2.5)
    movers['end_slope'] = np.clip(movers.end_speed * duration / movers.geometry_length, 0.05, 2.5)
    movers['motion_profile'] = 'hybrid'
    return movers

def infer_network_observed_link_times(peaks, regularization=OBSERVED_SPEED_REGULARIZATION):
    """Infer one effective peak travel time per link from all adjacent peak pairs."""
    link_ids = peaks.parent_link_id.to_numpy()
    index = {link_id: i for i, link_id in enumerate(link_ids)}
    equations, observations = [], []
    for source in peaks.itertuples(index=False):
        for target in peaks.loc[peaks.id_us_node.eq(source.id_ds_node)].itertuples(index=False):
            delay = target.timing_time_seconds - source.timing_time_seconds
            if delay <= 0:
                continue
            row = np.zeros(len(peaks))
            row[index[source.parent_link_id]] = 1.0
            row[index[target.parent_link_id]] = 1.0
            equations.append(row)
            observations.append(2.0 * delay)
    if not equations:
        raise ValueError('No positive adjacent peak-time delays available for observed-speed inference.')
    # Peak equations can be underdetermined on an alternating branch graph.
    # A weak K-based anchor selects a stable, positive effective link time.
    anchor = peaks.celerity_link_travel_seconds.to_numpy(dtype=float)
    design = np.vstack([np.vstack(equations), np.sqrt(regularization) * np.eye(len(peaks))])
    rhs = np.concatenate([np.asarray(observations), np.sqrt(regularization) * anchor])
    inferred = np.linalg.lstsq(design, rhs, rcond=None)[0]
    timestep = float(np.median(np.diff(time_seconds)))
    result = peaks.copy()
    result['network_observed_link_travel_seconds'] = np.maximum(inferred, timestep)
    return result

def individual_link_segments(peaks, travel_time_column):
    """Move independently along every full parent link, centred on its peak."""
    rows = []
    for link in peaks.itertuples(index=False):
        travel_seconds = max(float(getattr(link, travel_time_column)), 1.0)
        rows.append({'start_seconds': link.timing_time_seconds - travel_seconds / 2,
                     'end_seconds': link.timing_time_seconds + travel_seconds / 2,
                     'geometry': as_single_line(link.geometry),
                     'start_peak_width_seconds': link.peak_width_seconds,
                     'peak_width_seconds': link.peak_width_seconds})
    return pd.DataFrame(rows)

def build_movers(peaks, motion_mode):
    if motion_mode == 'midpoint_timing':
        return midpoint_timing_segments(peaks)
    if motion_mode == 'hybrid_midpoint_timing':
        return hybrid_midpoint_segments(peaks)
    if motion_mode == 'observed_link_speed':
        return individual_link_segments(peaks, 'observed_link_travel_seconds')
    if motion_mode == 'network_observed_link_speed':
        inferred = infer_network_observed_link_times(peaks)
        return individual_link_segments(inferred, 'network_observed_link_travel_seconds')
    if motion_mode == 'celerity':
        return individual_link_segments(peaks, 'celerity_link_travel_seconds')
    raise ValueError("MOTION_MODE must be 'midpoint_timing', 'hybrid_midpoint_timing', 'observed_link_speed', 'network_observed_link_speed', or 'celerity'.")



# Shared rendering helpers — required even when the side-by-side GIF is disabled.
import matplotlib.pyplot as plt
from PIL import Image
from tempfile import TemporaryDirectory

all_bounds = np.array([
    min(base_peaks.total_bounds[0], variant_peaks.total_bounds[0]),
    min(base_peaks.total_bounds[1], variant_peaks.total_bounds[1]),
    max(base_peaks.total_bounds[2], variant_peaks.total_bounds[2]),
    max(base_peaks.total_bounds[3], variant_peaks.total_bounds[3]),
])

def draw_scale_bar(ax):
    """A map-coordinate scale, fixed in the lower-right corner of every frame."""
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    crs = base_peaks.crs
    if crs is None or not crs.is_projected:
        raise ValueError('The scale bar requires projected map coordinates.')
    metres_per_unit = float(crs.axis_info[0].unit_conversion_factor)
    width = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    if SCALE_BAR_KM is None:
        target_km = width * metres_per_unit * 0.30 / 1000
        power = 10 ** np.floor(np.log10(target_km))
        length_km = max(value * power for value in [1, 2, 5] if value * power <= target_km)
    else:
        length_km = float(SCALE_BAR_KM)
    if not np.isfinite(length_km) or length_km <= 0:
        raise ValueError('SCALE_BAR_KM must be a positive length or None.')
    length_units = length_km * 1000 / metres_per_unit
    if length_units > width * 0.85:
        raise ValueError('Scale bar is too wide for this map; reduce SCALE_BAR_KM or use None.')
    bar = AnchoredSizeBar(ax.transData, length_units, f'{length_km:g} km',
                         loc='lower right', pad=0.5, borderpad=0.7, sep=4,
                         size_vertical=abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.002,
                         color='#263238', frameon=True, fontproperties={'size': 10})
    bar.patch.set_facecolor('white')
    bar.patch.set_edgecolor('none')
    bar.patch.set_alpha(0.95)
    bar.set_zorder(100)
    ax.add_artist(bar)
    return bar

def mover_fraction(mover, current_time):
    linear_fraction = (current_time - mover.start_seconds) / (mover.end_seconds - mover.start_seconds)
    if getattr(mover, 'motion_profile', None) != 'hybrid':
        return float(linear_fraction)
    # Cubic Hermite timing: exact endpoints/peak times, continuous speed at
    # their shared waypoint, and no positional overshoot.
    u = float(linear_fraction)
    return float((u**3 - 2*u**2 + u) * mover.start_slope + (-2*u**3 + 3*u**2) + (u**3 - u**2) * mover.end_slope)

def mover_points(movers, current_time):
    active = movers.loc[movers.start_seconds.le(current_time) & movers.end_seconds.ge(current_time)]
    points = []
    for mover in active.itertuples(index=False):
        points.append(mover.geometry.interpolate(mover_fraction(mover, current_time), normalized=True))
    return points

def draw_movers(ax, links, movers, current_time, *, title):
    links.plot(ax=ax, color='#9aa0a6', linewidth=1.0, zorder=1)
    points = mover_points(movers, current_time)
    if points:
        ax.scatter([point.x for point in points], [point.y for point in points], s=DOT_SIZE, c='#d62728', edgecolors='white', linewidths=0.45, zorder=2)
    ax.set(xlim=(all_bounds[0], all_bounds[2]), ylim=(all_bounds[1], all_bounds[3]), title=title, aspect='equal')
    ax.set_axis_off()
    draw_scale_bar(ax)



import numpy as np

from PIL import Image

from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from matplotlib.collections import LineCollection

from matplotlib.colors import to_rgba

from shapely.ops import unary_union

if 'BASE_MOVER_STYLE' not in globals():
    BASE_MOVER_STYLE = 'dot'

if 'VARIANT_MOVER_STYLE' not in globals():
    VARIANT_MOVER_STYLE = 'dot'

if 'WAVE_STYLE' not in globals():
    WAVE_STYLE = 'dot'

if 'PULSE_MIN_LINK_FRACTION' not in globals():
    PULSE_MIN_LINK_FRACTION = 0.05

if 'PULSE_MAX_LINK_FRACTION' not in globals():
    PULSE_MAX_LINK_FRACTION = 0.34

if 'PULSE_LINEWIDTH' not in globals():
    PULSE_LINEWIDTH = 6.0

if 'PULSE_ALPHA' not in globals():
    PULSE_ALPHA = 0.28

if 'PULSE_OVERLAP_TOLERANCE_M' not in globals():
    PULSE_OVERLAP_TOLERANCE_M = 250.0

if 'SHARED_PULSE_COLOR' not in globals():
    SHARED_PULSE_COLOR = '#7c8790'

if 'VARIANT_DASH_PATTERN' not in globals():
    VARIANT_DASH_PATTERN = (0, (4, 2))

if 'KART_SPRITE_ZOOM' not in globals():
    KART_SPRITE_ZOOM = 0.020

SPRITE_DIR = REPO / 'RAPID/assets/peak_dot_sprites'

BASE_KART_SPRITE = np.asarray(Image.open(SPRITE_DIR / 'base_red_kart.png').convert('RGBA'))

VARIANT_KART_SPRITE = np.asarray(Image.open(SPRITE_DIR / 'variant_green_kart.png').convert('RGBA'))

if BASE_MOVER_STYLE not in {'dot', 'sprite'} or VARIANT_MOVER_STYLE not in {'dot', 'sprite'}:
    raise ValueError("Mover styles must be 'dot' or 'sprite'.")

def draw_sprite_points(ax, points, sprite, *, zoom, zorder):
    for point in points:
        artist = AnnotationBbox(OffsetImage(sprite, zoom=zoom), (point.x, point.y), frameon=False, pad=0, zorder=zorder)
        ax.add_artist(artist)

if WAVE_STYLE not in {'dot', 'peak_plus_width', 'hydro_pulse', 'hydro_pulse_dots', 'spatial_wave_field', 'reference_peak_wave'}:
    raise ValueError("Unknown WAVE_STYLE. Use 'dot', 'peak_plus_width', 'hydro_pulse', 'hydro_pulse_dots', 'spatial_wave_field', or 'reference_peak_wave'.")

_pulse_widths = np.concatenate([
    base_peaks.peak_width_seconds.to_numpy(),
    variant_peaks.peak_width_seconds.to_numpy(),
])

_pulse_width_low, _pulse_width_high = np.quantile(_pulse_widths, [0.10, 0.90])

_pulse_width_span = max(float(_pulse_width_high - _pulse_width_low), 1.0)

_pulse_reference_length = float(np.median(np.concatenate([
    base_peaks.geometry.map(lambda geometry: geometry.length).to_numpy(),
    variant_peaks.geometry.map(lambda geometry: geometry.length).to_numpy(),
])))

def active_movers(movers, current_time):
    active = movers.loc[movers.start_seconds.le(current_time) & movers.end_seconds.ge(current_time)].copy()
    active['fraction'] = [mover_fraction(mover, current_time) for mover in active.itertuples(index=False)]
    return active

def draw_spatial_wave_field(ax, field, current_time, *, color, zorder):
    """Render the local RAPID hydrograph on every subreach at this time."""
    upper = int(np.searchsorted(time_seconds, current_time, side='right'))
    lower = max(0, upper - 1)
    upper = min(upper, len(time_seconds) - 1)
    if upper == lower:
        discharge = field['qout'][lower]
    else:
        weight = (current_time - time_seconds[lower]) / (time_seconds[upper] - time_seconds[lower])
        discharge = (1.0 - weight) * field['qout'][lower] + weight * field['qout'][upper]
    intensity = np.clip((discharge - field['baseline']) / field['scale'], 0.0, 1.0)[field['columns']]
    rgba = np.tile(to_rgba(color), (len(field['paths']), 1))
    rgba[:, 3] = 0.02 + 0.90 * intensity
    widths = 0.5 + 5.5 * intensity
    ax.add_collection(LineCollection(field['paths'], colors=rgba, linewidths=widths, capstyle='round', zorder=zorder))

def pulse_sections_across_midpoints(mover, movers, *, fraction, half_distance, max_hops=3):
    """Return pulse pieces continued across adjacent midpoint segments."""
    line_length = mover.geometry.length
    centre_distance = fraction * line_length
    sections = [substring(
        mover.geometry,
        max(0.0, centre_distance - half_distance),
        min(line_length, centre_distance + half_distance),
    )]

    def extend_backward(link_id, remaining, hops):
        if remaining <= 0 or hops <= 0 or pd.isna(link_id):
            return
        for previous in movers.loc[movers.end_link_id.eq(link_id)].itertuples(index=False):
            length = previous.geometry.length
            take = min(remaining, length)
            sections.append(substring(previous.geometry, length - take, length))
            if remaining > length:
                extend_backward(previous.start_link_id, remaining - length, hops - 1)

    def extend_forward(link_id, remaining, hops):
        if remaining <= 0 or hops <= 0 or pd.isna(link_id):
            return
        for following in movers.loc[movers.start_link_id.eq(link_id)].itertuples(index=False):
            length = following.geometry.length
            take = min(remaining, length)
            sections.append(substring(following.geometry, 0, take))
            if remaining > length:
                extend_forward(following.end_link_id, remaining - length, hops - 1)

    extend_backward(mover.start_link_id, half_distance - centre_distance, max_hops)
    extend_forward(mover.end_link_id, centre_distance + half_distance - line_length, max_hops)
    return [section for section in sections if not section.is_empty and section.length > 0]

def local_pulse_half_distance(mover, current_time):
    """A visually stable, midpoint-continuous pulse length."""
    time_fraction = (current_time - mover.start_seconds) / (mover.end_seconds - mover.start_seconds)
    width_seconds = (1.0 - time_fraction) * mover.start_peak_width_seconds + time_fraction * mover.peak_width_seconds
    width_fraction = np.clip((width_seconds - _pulse_width_low) / _pulse_width_span, 0.0, 1.0)
    full_fraction = PULSE_MIN_LINK_FRACTION + (PULSE_MAX_LINK_FRACTION - PULSE_MIN_LINK_FRACTION) * width_fraction
    return 0.5 * full_fraction * _pulse_reference_length

def pulse_geometry(movers, current_time):
    sections = []
    for mover in active_movers(movers, current_time).itertuples(index=False):
        sections.extend(pulse_sections_across_midpoints(
            mover, movers, fraction=mover.fraction,
            half_distance=local_pulse_half_distance(mover, current_time),
        ))
    return unary_union(sections) if sections else None

def draw_pulse_geometry(ax, geometry, *, color, zorder):
    if geometry is None or geometry.is_empty:
        return
    parts = [geometry] if geometry.geom_type == 'LineString' else [part for part in geometry.geoms if part.geom_type == 'LineString']
    if parts:
        ax.add_collection(LineCollection([np.asarray(part.coords) for part in parts], colors=[to_rgba(color, alpha=PULSE_ALPHA)], linewidths=PULSE_LINEWIDTH, capstyle='round', zorder=zorder))

def draw_hydro_pulse_dots(ax, movers, current_time, *, color, dot_size, zorder):
    """Large peak dots with smaller rising and receding shoulder dots."""
    active = active_movers(movers, current_time)
    peak_points, shoulder_points = [], []
    for mover in active.itertuples(index=False):
        duration = max(float(mover.end_seconds - mover.start_seconds), 1.0)
        width_seconds = (1.0 - mover.fraction) * mover.start_peak_width_seconds + mover.fraction * mover.peak_width_seconds
        # Scale pulse length to the measured local hydrograph width.
        offset = np.clip(0.5 * width_seconds / duration, 0.035, 0.18)
        peak_points.append(mover.geometry.interpolate(float(mover.fraction), normalized=True))
        for fraction in (mover.fraction - offset, mover.fraction + offset):
            if 0.0 <= fraction <= 1.0:
                shoulder_points.append(mover.geometry.interpolate(float(fraction), normalized=True))
    if shoulder_points:
        ax.scatter([point.x for point in shoulder_points], [point.y for point in shoulder_points], s=dot_size * 0.52, c=color, alpha=0.65, edgecolors='white', linewidths=0.3, zorder=zorder)
    if peak_points:
        ax.scatter([point.x for point in peak_points], [point.y for point in peak_points], s=dot_size, c=color, edgecolors='white', linewidths=0.45, zorder=zorder + 1)

def draw_mover_layer(ax, movers, current_time, *, color, style, sprite, zoom, marker, dot_size, zorder, draw_pulse=True):
    active = active_movers(movers, current_time)
    if active.empty:
        return
    if WAVE_STYLE == 'hydro_pulse_dots':
        draw_hydro_pulse_dots(ax, movers, current_time, color=color, dot_size=dot_size, zorder=zorder)
        return
    if draw_pulse and WAVE_STYLE in {'peak_plus_width', 'hydro_pulse'}:
        draw_pulse_geometry(ax, pulse_geometry(movers, current_time), color=color, zorder=zorder)
    if WAVE_STYLE == 'hydro_pulse':
        return
    points = [mover.geometry.interpolate(float(mover.fraction), normalized=True) for mover in active.itertuples(index=False)]
    if style == 'sprite':
        draw_sprite_points(ax, points, sprite, zoom=zoom, zorder=zorder + 1)
    else:
        ax.scatter([point.x for point in points], [point.y for point in points], marker=marker, s=dot_size, c=color, edgecolors='white', linewidths=0.45, zorder=zorder + 1)

def draw_overlay(ax, current_time):
    if WAVE_STYLE == 'reference_peak_wave':
        # Reference-only: blue base geometry and a red travelling peak front.
        base_peaks.plot(ax=ax, color='#1976d2', linewidth=1.45, alpha=0.78, zorder=1)
        draw_mover_layer(ax, base_movers, current_time, color='#d62728', style='dot', sprite=None, zoom=0, marker='o', dot_size=DOT_SIZE, zorder=3, draw_pulse=False)
        ax.set(xlim=(all_bounds[0], all_bounds[2]), ylim=(all_bounds[1], all_bounds[3]), aspect='equal')
        ax.set_axis_off()
        draw_scale_bar(ax)
        return
    # Keep the full base; draw variant geometry only within the selected collapse boundaries.
    variant_overlay_peaks.plot(ax=ax, color='#474D4F', linewidth=1.3, linestyle=VARIANT_DASH_PATTERN, alpha=0.95, zorder=2)
    base_peaks.plot(ax=ax, color='#B2BBC2', linewidth=1.35, zorder=1)
    if WAVE_STYLE == 'spatial_wave_field':
        # This is not a travelling icon: it is RAPID's local discharge state
        # along every subreach, so width and intensity remain physically local.
        draw_spatial_wave_field(ax, variant_overlay_wave_field, current_time, color='#1976d2', zorder=5)
        draw_spatial_wave_field(ax, base_wave_field, current_time, color='#d62728', zorder=3)
    elif WAVE_STYLE in {'peak_plus_width', 'hydro_pulse'}:
        variant_pulse = pulse_geometry(variant_movers, current_time)
        base_pulse = pulse_geometry(base_movers, current_time)
        if base_pulse is not None and variant_pulse is not None:
            # Exact line equality is too strict: tiny timing/width differences
            # create distracting coloured fragments. Use the base centreline as
            # the neutral shared front wherever the variant is nearby.
            base_near_variant = base_pulse.intersection(variant_pulse.buffer(PULSE_OVERLAP_TOLERANCE_M))
            base_only = base_pulse.difference(variant_pulse.buffer(PULSE_OVERLAP_TOLERANCE_M))
            variant_only = variant_pulse.difference(base_pulse.buffer(PULSE_OVERLAP_TOLERANCE_M))
        else:
            base_near_variant, base_only, variant_only = None, base_pulse, variant_pulse
        draw_pulse_geometry(ax, variant_only, color='#1976d2', zorder=3)
        draw_pulse_geometry(ax, base_near_variant, color=SHARED_PULSE_COLOR, zorder=4)
        draw_pulse_geometry(ax, base_only, color='#d62728', zorder=5)
        if WAVE_STYLE == 'peak_plus_width':
            draw_mover_layer(ax, variant_movers, current_time, color='#1976d2', style=VARIANT_MOVER_STYLE, sprite=VARIANT_KART_SPRITE, zoom=KART_SPRITE_ZOOM, marker='^', dot_size=DOT_SIZE * 1.55, zorder=6, draw_pulse=False)
            draw_mover_layer(ax, base_movers, current_time, color='#d62728', style=BASE_MOVER_STYLE, sprite=BASE_KART_SPRITE, zoom=KART_SPRITE_ZOOM, marker='o', dot_size=DOT_SIZE, zorder=7, draw_pulse=False)
    else:
        draw_mover_layer(ax, variant_movers, current_time, color='#1976d2', style=VARIANT_MOVER_STYLE, sprite=VARIANT_KART_SPRITE, zoom=KART_SPRITE_ZOOM, marker='^', dot_size=DOT_SIZE * 1.55, zorder=3)
        draw_mover_layer(ax, base_movers, current_time, color='#d62728', style=BASE_MOVER_STYLE, sprite=BASE_KART_SPRITE, zoom=KART_SPRITE_ZOOM, marker='o', dot_size=DOT_SIZE, zorder=5)
    ax.set(xlim=(all_bounds[0], all_bounds[2]), ylim=(all_bounds[1], all_bounds[3]), aspect='equal')
    ax.set_axis_off()
    draw_scale_bar(ax)

MOTION_MODES_TO_SAVE = globals().get('MOTION_MODES_TO_SAVE', [MOTION_MODE])
WAVE_STYLES_TO_SAVE = globals().get('WAVE_STYLES_TO_SAVE', ['dot'])
base_movers = build_movers(base_peaks, MOTION_MODE)
variant_movers = build_movers(variant_peaks, MOTION_MODE)

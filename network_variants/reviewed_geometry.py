"""Versioned, reviewed geometry inputs shared by preparation and regeneration.

Geometry acceptance is separate from hydraulic readiness. Original experiment
products remain available for comparison; an active release selects replacements
and canonical independent sources without mixing old attributes into new links.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / 'network_variants/config/reviewed_geometry.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ReviewedGeometry:
    def __init__(self, manifest=None):
        if manifest is None:
            if not ACTIVE.exists():
                self.manifest = None
                self.records = {}
                return
            pointer = json.loads(ACTIVE.read_text())
            manifest = Path(pointer['manifest'])
            if digest(manifest) != pointer['sha256']:
                raise ValueError('Active reviewed-geometry manifest checksum mismatch')
        self.manifest = Path(manifest)
        self._source_node_index = None
        data = json.loads(self.manifest.read_text())
        self.release = data['release']
        self.hydraulic_regeneration_complete = data.get('hydraulic_regeneration_complete', False)
        self.records = {(r['example'], r['family'], r['state']): r for r in data['records']}
        if len(self.records) != len(data['records']):
            raise ValueError('Duplicate logical states in geometry release')

    def lookup(self, example, family, state):
        return self.records.get((str(example).zfill(2), family, state))

    def canonical(self, record):
        seen = set()
        while record.get('canonical_key'):
            key = tuple(record['canonical_key'])
            if key in seen:
                raise ValueError('Cycle in canonical geometry aliases')
            seen.add(key)
            record = self.records[key]
        return record

    def load(self, example, family, state):
        """Return geometry, saved parent identities and reviewed domain together."""
        import geopandas as gpd
        record = self.lookup(example, family, state)
        if record is None:
            raise KeyError((example, family, state))
        canonical = self.canonical(record)
        if canonical.get('geometry_status') == 'excluded':
            raise ValueError('State is excluded from the integrated geometry population')
        bundle = canonical.get('bundle')
        if bundle:
            folder = Path(bundle)
            meta = json.loads((folder / 'bundle.json').read_text())
            core = {name: meta['checksums'][name] for name in
                    ['links.parquet','nodes.parquet','contract.json','validated_node_correspondences.json']}
            identity = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:24]
            if identity != canonical['geometry_id']:
                raise ValueError('Geometry bundle identity mismatch')
            for name, expected in meta['checksums'].items():
                if digest(folder / name) != expected:
                    raise ValueError(f'Reviewed geometry changed: {folder / name}')
            links = gpd.read_parquet(folder / 'links.parquet')
            nodes = gpd.read_parquet(folder / 'nodes.parquet')
            contract = json.loads((folder / 'contract.json').read_text())
            matches = {int(k): int(v) for k, v in json.loads((folder / 'validated_node_correspondences.json').read_text()).items()}
        else:
            for path, expected in canonical.get('retained_source_checksums', {}).items():
                if digest(path) != expected:
                    raise ValueError(f'Retained geometry/identity source changed: {path}')
            links = gpd.read_file(canonical['links'])
            nodes = gpd.read_file(canonical['nodes'])
            contract, matches = {}, {}
            if canonical.get('validated_correspondences_path'):
                matches = {int(k): int(v) for k, v in json.loads(Path(canonical['validated_correspondences_path']).read_text()).items()}
            # Unchanged geometry still needs its original canonical node IDs.
            # Do not rediscover these by proximity during hydraulic regeneration.
            if canonical.get('original_frame'):
                import pandas as pd
                frame = pd.read_parquet(canonical['original_frame'])
                pairs = pd.concat([
                    frame[['id_us_node','u']].rename(columns={'id_us_node':'raw','u':'parent'}),
                    frame[['id_ds_node','v']].rename(columns={'id_ds_node':'raw','v':'parent'})
                ]).drop_duplicates()
                if pairs.raw.duplicated().any() or pairs.parent.duplicated().any():
                    raise ValueError('Original node identities are not injective')
                mapping = dict(zip(pairs.raw.astype(int), pairs.parent.astype(int)))
                nodes['parent_identity'] = nodes.id_node.map(mapping).astype('Int64')
                if not (set(links.id_us_node)|set(links.id_ds_node)) <= set(mapping):
                    raise ValueError('Unchanged geometry differs from its original identity frame')
                links['u'] = links.id_us_node.map(mapping)
                links['v'] = links.id_ds_node.map(mapping)
            if canonical.get('contract_path'):
                path=Path(canonical['contract_path'])
                if digest(path) != canonical['contract_sha256']:
                    raise ValueError('Reviewed boundary contract checksum mismatch')
                contract=json.loads(path.read_text())
        return links, nodes, contract, matches, record

    def assert_hydraulics_current(self, inventory):
        """Reject old hydraulic inventories for changed geometry or canonical states.

A regenerated inventory must record the geometry release and geometry ID on
all affected rows, and the release must mark hydraulic regeneration complete.
Labels alone cannot activate unfinished hydraulic products. Separate duplicate
island observations must be excluded.
"""
        stale = []
        for row in inventory.to_dict('records'):
            record = self.lookup(row['example'], row['family'], row['state'])
            if record is None:
                if any(k[0] == str(row['example']).zfill(2) for k in self.records):
                    stale.append(f"{row['example']}/{row['state']}: state is not in the reviewed registry")
                continue
            if record.get('geometry_status') == 'excluded':
                if row.get('status') != 'excluded_topology_review':
                    stale.append(f"{record['example']}/{record['state']}: excluded state is marked routable")
                continue
            if record.get('separate_observation_excluded'):
                if row.get('status') != 'excluded_duplicate':
                    stale.append(f"{record['example']}/{record['state']}: duplicate island must use independent canonical state")
                continue
            canonical = self.canonical(record)
            if canonical.get('master_products_published') and row.get('status') == 'ready':
                frame = row.get('frame')
                expected = canonical['original_frame']
                if not frame or Path(frame).resolve() != Path(expected).resolve() or row.get('fingerprint') != canonical['original_fingerprint']:
                    stale.append(f"{record['example']}/{record['state']}: not the published master hydraulic frame")
            if canonical.get('requires_hydraulic_regeneration') and (
                not self.hydraulic_regeneration_complete or
                row.get('geometry_release') != self.release or
                row.get('geometry_id') != canonical.get('geometry_id') or
                row.get('status') != 'ready'
            ):
                stale.append(f"{record['example']}/{record['state']}")
        if stale:
            raise ValueError(f'{len(stale)} states use hydraulic inputs predating the integrated geometry. '
                             f'Regenerate attributes and preparation first. Examples: {", ".join(stale[:5])}')

    def assert_preparation_current(self, context):
        """Prevent normal preparation from pairing old widths with repaired nodes."""
        if not self.records:
            return
        source = str(Path(context.directed_nodes_path).resolve())
        if self._source_node_index is None:
            self._source_node_index = {}
            for record in self.records.values():
                if record.get('nodes'):
                    path = str(Path(record['nodes']).resolve())
                    self._source_node_index.setdefault(path, []).append(record)
                canonical = self.canonical(record)
                if canonical.get('bundle'):
                    path = str((Path(canonical['bundle'])/'nodes.parquet').resolve())
                    self._source_node_index.setdefault(path, []).append(record)
        matches = self._source_node_index.get(source, [])
        checked = set()
        for record in matches:
            canonical = self.canonical(record)
            key = (canonical['example'], canonical['family'], canonical['state'])
            if key in checked:
                continue
            checked.add(key)
            if canonical.get('master_products_published'):
                if Path(context.link_widths_path).resolve() != Path(canonical['links']).resolve():
                    raise ValueError('Preparation must use the published master width products.')
                if Path(context.hydraulic_context_output_dir).resolve() != Path(canonical['original_manifest']).resolve().parent:
                    raise ValueError('Preparation must use the published master hydraulic context.')
                for path, expected in canonical.get('retained_source_checksums', {}).items():
                    stat = Path(path).stat()
                    if _cached_digest(path, stat.st_mtime_ns, stat.st_size) != expected:
                        raise ValueError(f'Published master product changed: {path}')
            if canonical.get('requires_hydraulic_regeneration') and (
                not self.hydraulic_regeneration_complete or
                getattr(context, 'geometry_release', None) != self.release or
                getattr(context, 'geometry_id', None) != canonical.get('geometry_id')
            ):
                raise ValueError(f"{record['example']}/{record['state']} has integrated geometry changes. "
                                 "Regenerate hydraulic products before RAPID preparation; "
                                 "the current widths/heads belong to the old geometry.")

    def annotate_state_registry(self, experiment, frame):
        """Expose integrated geometry/canonical identities to normal preparation.

        Existing width/head paths are retained as provenance until regenerated;
        assert_preparation_current prevents those old products being routed.
        """
        import re
        match = re.match(r'^(\d+)_(indep|compound_bubbles|island)(?:_|$)', Path(experiment).name)
        if not match or not self.records:
            return frame
        example, family = match.groups()
        frame = frame.copy()
        for idx, row in frame.iterrows():
            record = self.lookup(example, family, row['state_id'])
            if record is None:
                continue
            canonical = self.canonical(record)
            folder = Path(canonical['bundle']) if canonical.get('bundle') else None
            frame.loc[idx, 'integrated_geometry_release'] = self.release
            frame.loc[idx, 'integrated_geometry_id'] = canonical.get('geometry_id', '')
            frame.loc[idx, 'integrated_geometry_links'] = str(folder/'links.parquet') if folder else canonical.get('links','')
            frame.loc[idx, 'integrated_geometry_nodes'] = str(folder/'nodes.parquet') if folder else canonical.get('nodes','')
            frame.loc[idx, 'integrated_canonical_family'] = canonical['family']
            frame.loc[idx, 'integrated_canonical_state'] = canonical['state']
            frame.loc[idx, 'integrated_geometry_status'] = record['geometry_status']
            frame.loc[idx, 'integrated_hydraulics_pending'] = canonical.get('requires_hydraulic_regeneration',False) and not self.hydraulic_regeneration_complete
        return frame


from functools import lru_cache


@lru_cache(maxsize=64000)
def _cached_digest(path, modified_ns, size):
    return digest(path)


@lru_cache(maxsize=4)
def _cached_active(path, modified_ns, size):
    return ReviewedGeometry()


def active_geometry():
    """Cache a release across preparation calls, invalidating on pointer change."""
    if not ACTIVE.exists():
        return ReviewedGeometry()
    stat = ACTIVE.stat()
    return _cached_active(str(ACTIVE), stat.st_mtime_ns, stat.st_size)

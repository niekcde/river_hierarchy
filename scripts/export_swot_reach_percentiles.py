"""Node temporal percentiles -> equally weighted reach means.

Requires duckdb. Source files are read only. Run with --help for paths.
"""
from pathlib import Path
import argparse
import tempfile
import duckdb


def export(node_files, sword_files, output, memory='4GB', temp_dir=None, batches=128):
    """Stage narrow columns once; keep each reach entirely in one disk batch."""
    if batches < 1:
        raise ValueError('batches must be positive')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(temp_dir) if temp_dir else output.parent / '.swot_tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='percentiles-', dir=scratch) as work:
        con = duckdb.connect()
        try:
            _export(con, node_files, sword_files, output, memory, Path(work), batches)
        finally:
            con.close()
    print(f'Saved {output}')


def _export(con, node_files, sword_files, output, memory, work, batches):
    con.execute('SET memory_limit = ?', [memory])
    con.execute('SET threads = 2')
    con.execute('SET preserve_insertion_order = false')
    con.execute('SET temp_directory = ?', [str(work / 'spill')])
    con.from_parquet([str(p) for p in node_files], union_by_name=True,
                     filename=True).create_view('source_nodes')
    con.from_parquet([str(p) for p in sword_files], union_by_name=True).create_view('sword')
    def literal(path):
        return "'" + str(path).replace("'", "''") + "'"
    # Project only required fields, avoiding a global sort of the full archive.
    # All versions and visits of a reach land in the same partition.
    print('Staging required columns into disk batches (one archive scan)...', flush=True)
    con.execute(f"""
        COPY (
            SELECT CAST(reach_id AS BIGINT) AS reach_id,
                   CAST(node_id AS BIGINT) AS node_id,
                   cycle, pass, time_str, TRY_CAST(version AS INTEGER) AS version,
                   filename AS source_filename,
                   wse_sm, width, wse_sm_u, dark_frac, xtrk_dist,
                   hash(CAST(reach_id AS BIGINT)) % {int(batches)} AS bucket
            FROM source_nodes
            WHERE TRY_CAST(time_str AS TIMESTAMP) IS NOT NULL
              AND cycle IS NOT NULL AND pass IS NOT NULL
        ) TO {literal(work / 'staged')}
        (FORMAT PARQUET, PARTITION_BY (bucket), COMPRESSION ZSTD)
    """)
    con.execute("""CREATE TEMP TABLE reach_stats (
        reach_id BIGINT, wse_p10 DOUBLE, wse_p50 DOUBLE, wse_p90 DOUBLE,
        width_p10 DOUBLE, width_p50 DOUBLE, width_p90 DOUBLE,
        n_nodes BIGINT, n_observations BIGINT
    )""")
    partitions = sorted((work / 'staged').glob('bucket=*'))
    for index, partition in enumerate(partitions, 1):
        print(f'Processing batch {index}/{len(partitions)}', flush=True)
        con.from_parquet(str(partition / '*.parquet'), hive_partitioning=False).create_view('raw_nodes', replace=True)
        _process_batch(con)
        con.execute('DROP VIEW raw_nodes')
        # Free completed batches immediately; only this run's scratch files.
        for f in partition.glob('*.parquet'):
            f.unlink()
        partition.rmdir()
    result = con.sql("""
        WITH inventory AS (
            SELECT DISTINCT CAST(reach_id AS BIGINT) AS reach_id FROM sword
        )
        SELECT i.reach_id,
               r.wse_p10, r.wse_p50, r.wse_p90,
               r.width_p10, r.width_p50, r.width_p90,
               coalesce(r.n_nodes, 0) AS n_nodes,
               coalesce(r.n_observations, 0) AS n_observations
        FROM inventory i LEFT JOIN reach_stats r USING (reach_id)
        ORDER BY i.reach_id
    """)
    result.write_csv(str(output), header=True)


def _process_batch(con):
    # Deduplicate before screening: highest numeric file version wins;
    # filename breaks ties deterministically. One observation per node/cycle/pass.
    con.execute('''
        CREATE OR REPLACE TEMP TABLE node_stats AS
        WITH deduplicated AS (
            SELECT * FROM raw_nodes
            WHERE TRY_CAST(time_str AS TIMESTAMP) IS NOT NULL
              AND cycle IS NOT NULL AND pass IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY CAST(reach_id AS BIGINT), CAST(node_id AS BIGINT), cycle, pass
                ORDER BY TRY_CAST(version AS INTEGER) DESC NULLS LAST,
                         source_filename DESC, time_str DESC
            ) = 1
        ), screened AS (
            SELECT CAST(reach_id AS BIGINT) AS reach_id,
                   CAST(node_id AS BIGINT) AS node_id, wse_sm, width
            FROM deduplicated
            WHERE wse_sm_u <= 1
              AND dark_frac <= 0.5
              AND ABS(xtrk_dist) BETWEEN 10000 AND 60000
              AND wse_sm > 0
              AND isfinite(wse_sm) AND isfinite(width) AND width > 0
        )
        SELECT reach_id, node_id,
               quantile_cont(wse_sm, [0.1, 0.5, 0.9]) AS wse_q,
               quantile_cont(width, [0.1, 0.5, 0.9]) AS width_q,
               count(*) AS n_observations
        FROM screened
        GROUP BY reach_id, node_id
    ''')
    con.execute("""
        INSERT INTO reach_stats
        SELECT reach_id,
               avg(wse_q[1]), avg(wse_q[2]), avg(wse_q[3]),
               avg(width_q[1]), avg(width_q[2]), avg(width_q[3]),
               count(*), CAST(sum(n_observations) AS BIGINT)
        FROM node_stats GROUP BY reach_id
    """)
    con.execute('DROP TABLE node_stats')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes', type=Path, default=Path('/Volumes/PhD/SWOT/RiverSP_D_parq/node'))
    p.add_argument('--sword', type=Path, default=Path('/Volumes/PhD/SWORD/v17c'))
    p.add_argument('--output', type=Path, default=Path('swot_reach_percentiles.csv'))
    p.add_argument('--memory', default='4GB')
    p.add_argument('--temp-dir', type=Path, help='Scratch disk for staged Parquet and DuckDB spill files')
    p.add_argument('--batches', type=int, default=128, help='More batches reduce peak sorting/aggregation memory')
    a = p.parse_args()
    nodes = sorted(f for f in a.nodes.glob('*.parquet') if not f.name.startswith('.'))
    sword = sorted(a.sword.glob('sword_*_v17c_reaches.parquet'))
    if not nodes or not sword:
        p.error('No input files found; check --nodes and --sword.')
    print(f'Reading {len(nodes)} SWOT files and {len(sword)} SWORD files.', flush=True)
    export(nodes, sword, a.output, a.memory, a.temp_dir, a.batches)


if __name__ == '__main__':
    main()

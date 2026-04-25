# Sensitivity Configs

Config-driven entry points now use these example manifests:

- `recipes.example.json`
- `grid.example.json`

The structural sensitivity recipe set remains the 7-record
`networks_sensitivity.jsonl.gz` fixture.

The separate single-edge baseline control now lives as the shared explicit
artifact `../../configs/single_edge_control.json`
and should be passed into the sensitivity workflow rather than injected as a
hidden runner special case.

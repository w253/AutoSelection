# Extension Example

`dummy_extension.py` shows the four optional extension surfaces:

```text
register_operators(registry)          # add executable operators
precompute_features(samples, context) # cold-start feature computation
RECIPE_HOOKS                          # observe recipe execution lifecycle
OPERATOR_CATALOG_PATCH                # add LLM prompt/catalog metadata
```

Run the smoke test:

```bash
PYTHONPATH=src:. python -m unittest tests.test_extensions_smoke
```

Try it in the E2E script:

```bash
EXTENSION_MODULES=examples.extensions.dummy_extension bash runs/run_mcts_e2e.sh
```

For real operators, keep the expensive feature computation in
`precompute_features()` and keep `transform()` focused on deterministic filtering
or mixing based on those cached metadata fields.

# `.github/workflows/`

CI for `gonka-poc`. All workflows run on `ubuntu-latest` and are CPU-only —
GPU validation happens downstream in the deployment pipeline.

## Workflows

### `contract-tests.yml`

Read-only API-drift detector against the `vllm` versions we target.

- **What it does.** Installs `vllm` (pinned per matrix entry), installs
  `gonka-poc` with the `[test]` extra, runs `pytest tests/contract tests/unit`. The
  contract suite only imports vllm internals and asserts the shapes
  (classes, signatures, attribute names) the plugin binds to. No model is
  loaded; no CUDA is touched.
- **Triggers.**
  - `push` to `main`
  - `pull_request` against any branch
  - `workflow_dispatch` (manual)
  - Skipped automatically when the diff is docs-only (`**/*.md`, `docs/**`,
    `LICENSE`).
- **Matrix.** `vllm == 0.23.0` and `vllm == 0.25.1` — published-on-PyPI
  versions only (vllm RC tags exist on GitHub but are never published to
  PyPI). Add a matrix entry when a new supported version ships.
- **Manual trigger.**
  - GitHub UI: *Actions → contract-tests → Run workflow → pick a ref*.
  - CLI: `gh workflow run contract-tests.yml --ref <branch>`.

#### `smoke-help` job (same workflow, same matrix)

Real-wheel composition smoke: verifies the `gonka-vllm-serve` console
script resolves to our entrypoint, imports the entrypoint/worker/`poc.*`
modules against the installed vllm wheel, and checks the
`vllm.general_plugins` entry-point loads and invokes idempotently.
Catches packaging and import-path bugs the contract suite cannot see.

### `grep-lint.yml`

Process gate running `tools/grep_lint.py` (pure stdlib, no install step).
Fails PRs that import `vllm.v1.*` outside `src/gonka_poc/_compat/` (the
only blessed channel for upstream internals) or cite an `ADR-NNNN` with
no matching file under `docs/adr/`.

## What to do if contract tests fail

A failing contract test means the vllm surface the plugin binds to changed.
The fix is **not** to relax the test — the test is the spec. Instead:

1. Read the failing assertion in `tests/contract/` to see exactly which
   vllm symbol drifted (renamed, signature changed, moved module, etc.).
2. Open `src/gonka_poc/_compat/` and add (or extend) a per-version compat
   module that adapts the new vllm surface to the shape the rest of
   `gonka_poc` expects. Convention:
   - `_compat/v0_23.py`, `_compat/v0_25.py` — one module per supported
     vllm minor.
   - `_compat/__init__.py` dispatches on `vllm.__version__`.
3. Update `pyproject.toml`'s `vllm>=X,<Y` bound only if the new range is
   actually supported by the compat layer.
4. Add a matrix entry in `contract-tests.yml` for the new vllm version so
   regressions are caught on the next push.

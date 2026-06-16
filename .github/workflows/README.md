# `.github/workflows/`

CI for `gonka-poc`. All workflows run on `ubuntu-latest` and are CPU-only —
GPU validation lives in the downstream `mlnode` pipelines.

## Workflows

### `contract-tests.yml`

Read-only API-drift detector against the `vllm` versions we target.

- **What it does.** Installs `vllm` (pinned per matrix entry), installs
  `gonka-poc` with the `[test]` extra, runs `pytest tests/contract`. The
  contract suite only imports vllm internals and asserts the shapes
  (classes, signatures, attribute names) the plugin binds to. No model is
  loaded; no CUDA is touched.
- **Triggers.**
  - `push` to `main`
  - `pull_request` against any branch
  - `workflow_dispatch` (manual)
  - Skipped automatically when the diff is docs-only (`**/*.md`, `docs/**`,
    `LICENSE`).
- **Matrix.** `vllm == 0.23.0` and `vllm == 0.23.1rc0`. The RC entry is an
  early-warning signal: if it fails, an upcoming vllm patch will break the
  plugin and we have a window to land a `_compat/` shim before the GA.
- **Manual trigger.**
  - GitHub UI: *Actions → contract-tests → Run workflow → pick a ref*.
  - CLI: `gh workflow run contract-tests.yml --ref <branch>`.

## What to do if contract tests fail

A failing contract test means the vllm surface the plugin binds to changed.
The fix is **not** to relax the test — the test is the spec. Instead:

1. Read the failing assertion in `tests/contract/` to see exactly which
   vllm symbol drifted (renamed, signature changed, moved module, etc.).
2. Open `src/gonka_poc/_compat/` and add (or extend) a per-version compat
   module that adapts the new vllm surface to the shape the rest of
   `gonka_poc` expects. Convention:
   - `_compat/vllm_0_23.py` — baseline.
   - `_compat/vllm_0_24.py` — for the next minor.
   - `_compat/__init__.py` dispatches on `vllm.__version__`.
3. Update `pyproject.toml`'s `vllm>=X,<Y` bound only if the new range is
   actually supported by the compat layer.
4. Add a matrix entry in `contract-tests.yml` for the new vllm version so
   regressions are caught on the next push.

If the RC entry (e.g. `0.23.1rc0`) is the only failing job, file an issue
against vllm linking the broken symbols before the GA cuts — they have a
window to revert or document the break.

# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `clrs/`, with most implementation details under `clrs/_src/` (algorithms, samplers, models, and the multi-solution framework in `clrs/_src/multi_sol/`).  
Entry points for training/evaluation are in `clrs/examples/`, especially `clrs/examples/run.py`.  

## Build, Test, and Development Commands
Run commands from the repository root.

- `python -m pip install -r requirements/requirements.txt` installs runtime dependencies.
- `python -m pip install -e .` installs the package in editable mode for local development.
- `python -m clrs.examples.run --train_steps 10000 --algorithms dfs_multi --hint_mode none --results_df --save_df --save_model_to_file --NSE 25 --test_length 16` runs a typical training/eval job.
- `python -m pytest clrs` runs the full test suite.
- `python -m pytest clrs/_src/multi_sol/core/manifest_loader_test.py` runs a focused test module while iterating.

Use `source clrs_env/bin/activate` for a pre-prepared environment.

## Coding Style & Naming Conventions
Python code follows the Google-style `pylintrc` in this repo.

- Use 2-space indentation (match existing files).
- Prefer `snake_case` for functions/variables/modules and `PascalCase` for classes.
- Keep module names lowercase with underscores (for example `manifest_loader.py`).
- Validate style with `pylint` when touching non-trivial logic: `pylint clrs`.

## Testing Guidelines
Use `pytest` as the runner; tests include both `absltest` and `unittest` test cases.  
Name new tests `*_test.py` and keep them near the code they cover.  
For multi-solution extensions, include registry/manifest coverage (see `clrs/_src/multi_sol/core/*_test.py`) and plugin-level tests under `clrs/_src/multi_sol/algorithms/<algo>/`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative, lowercase summaries (for example `move _multi algorithms...`, `enforce one source per level...`). Keep subject lines concise and specific.

For PRs:
- Describe behavior changes and rationale.
- Link the related issue/task.
- List exact verification commands run (tests/training command variants).
- Include sample output paths when results artifacts are part of the change.

Follow `CONTRIBUTING.md`: sign the Google CLA and use GitHub PR review workflow.

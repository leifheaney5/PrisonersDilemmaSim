# Contributing

## Pull request workflow

Keep each change based on the latest default branch and avoid opening several
pull requests that modify the same files for successive versions of one
feature. When a newer pull request supersedes an older one:

1. Rebase or recreate the replacement branch from the latest default branch.
2. Combine the complete intended change into the replacement pull request.
3. Confirm that the replacement contains every required commit and that CI is
   green.
4. Close the superseded pull requests instead of merging them in sequence.

This is especially important for `pages/app.py`, `pages/game_logic.py`, and
`pages/strategy_catalog.py`, where parallel feature branches are likely to edit
the same sections.

Before requesting review, run:

```bash
ruff check pages scripts tests
python -m unittest discover -s tests -v
python -m compileall -q pages scripts tests
git diff --check
```

Browser-facing changes should also run:

```bash
npm run test:e2e
```

## Resolving an existing conflict

Prefer a single replacement pull request when several unmerged pull requests
contain overlapping iterations of the same work. Start from the latest default
branch, apply the final combined state once, and use that replacement as the
only merge candidate. This avoids resolving the same conflict repeatedly and
prevents an older pull request from reverting a newer implementation.

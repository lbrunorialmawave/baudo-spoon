# Contributing to baudo-spoon

Thank you for taking the time to contribute.

## Commit Messages

All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) spec:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Types**: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`, `ci`
**Scopes**: `api`, `ml`, `scraper`, `db`, `docker`

**Examples**:
```
feat(ml): add SHAP explainability to XGBoost regression
fix(api): return 503 when Redis is unavailable instead of 500
chore(deps): bump scikit-learn to 1.5
```

## Branch Strategy

| Branch       | Purpose                          |
|--------------|----------------------------------|
| `main`       | Stable, production-ready code    |
| `improv`     | Integration branch for features  |
| `feat/<name>`| Feature branches (off `improv`)  |
| `fix/<name>` | Bug fix branches (off `main`)    |

## Pull Request Process

1. Fork and create a branch from `improv` (features) or `main` (hotfixes).
2. Ensure all CI checks pass.
3. Fill in the PR template completely.
4. Request a review from at least one CODEOWNER.
5. Squash and merge upon approval.

## Local Setup

See [README.md](README.md#getting-started) for the full setup guide.

## Code Style

Python code is linted and formatted with [ruff](https://docs.astral.sh/ruff/). Run before committing:

```bash
ruff check api/src ml scraper/src --fix
ruff format api/src ml scraper/src
```

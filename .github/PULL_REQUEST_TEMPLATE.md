## Summary

What does this change do and why?

## Changes

-

## Checklist

- [ ] `pre-commit install` done; hooks pass (or ran `pre-commit run --all-files`)
- [ ] Tests added/updated (and the `CubePick` mock exercises the change where applicable)
- [ ] Coverage stays at **100%** (`pytest --cov`)
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy` passes (strict)
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Public API changes are reflected in `roboinspect.__all__` and the API-snapshot test
- [ ] Core stays NumPy-only (new deps are optional extras, lazily imported)

## Related

Closes #

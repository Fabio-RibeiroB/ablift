# Changelog fragments

Don't edit `CHANGELOG.md` in a pull request. Add one small file here instead:

```text
changes/<issue>.<type>.md     e.g. changes/36.added.md
changes/+<slug>.<type>.md     for changes without an issue
```

`<type>` is one of `added`, `changed`, `fixed`, `removed` or `docs`. The file holds one or two sentences for users: what changed and why it matters.

At release time, `uv run towncrier build --version X.Y.Z --yes` moves every fragment into `CHANGELOG.md` and deletes the files. CI fails a pull request that changes `ablift/` without adding a fragment.

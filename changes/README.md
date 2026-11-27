# Changelog fragments

Don't edit `CHANGELOG.md` in a pull request. Add one small file here instead:

```text
changes/<issue>.<type>.md     e.g. changes/36.added.md
changes/+<slug>.<type>.md     for changes without an issue
```

`<type>` is one of `added`, `changed`, `fixed`, `removed` or `docs`. The file holds one or two sentences for users: what changed and why it matters.

At release time, `uv run towncrier build --version X.Y.Z --yes` moves every fragment into `CHANGELOG.md` and deletes the files. CI fails a pull request that changes `ablift/` without adding a fragment.

Run the same check locally with `make changelog-check` (it compares against `origin/main` and also runs as a pre-push hook). For internal changes with nothing to tell users, skip it with the `no-changelog` label on the pull request, or `SKIP_CHANGELOG=1` locally (e.g. `SKIP_CHANGELOG=1 git push`).

The release workflow refuses to publish a final (non pre-release) tag unless `CHANGELOG.md` has a `## [X.Y.Z]` section for it.

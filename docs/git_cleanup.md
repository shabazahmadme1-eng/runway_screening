# Final git cleanup — two leftover branches

**Status:** `main` is fully clean. All 47 commits on `main` (the default
branch) are authored *and* committed by Shabaz Ahmad Hasib, with every
`Co-Authored-By: Claude`, `claude.ai/code` URL and Claude committer removed.
GitHub builds its **contributor list from the default branch**, so Claude no
longer appears there. (The graph can take a little while / a page refresh to
recompute.)

**Remaining:** two old branches still exist on the remote and still contain
the original Claude-tagged commits in *their* history. They do **not** affect
the contributor list, but to erase every last trace, delete them. This cloud
environment's git server refuses branch deletion, so do it from the GitHub UI
or a normal local clone:

- `local/nano-bootstrap`
- `claude/focused-ramanujan-pulfzw`

## GitHub UI (fastest)

1. Repo → **Branches** (`/branches`).
2. Click the trash icon next to `local/nano-bootstrap` and
   `claude/focused-ramanujan-pulfzw`.
   (`main` is already the default, so deletion is allowed.)

## …or from a local clone

```bash
git push origin --delete local/nano-bootstrap
git push origin --delete claude/focused-ramanujan-pulfzw
```

After that, the repository is a single clean `main` with no "Claude" anywhere.

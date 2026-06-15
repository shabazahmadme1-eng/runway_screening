# Finishing the git cleanup (run locally)

The work was consolidated onto `main` from the cloud environment, but two
steps could not be done there because the environment's git proxy rejects
(a) branch deletions and (b) the large single push that a history rewrite
needs (HTTP 413 — it only accepts small incremental deltas). Run the
following from a normal local clone with full GitHub access.

## 1. Remove "Claude" from old commit history + squash

```bash
pip install git-filter-repo
git clone https://github.com/shabazahmadme1-eng/runway_screening.git
cd runway_screening

# scrub author/committer + Claude lines from every commit message (all branches)
git filter-repo --force \
  --name-callback  'return b"Shabaz Ahmad Hasib"' \
  --email-callback 'return b"shabazahmad.me1@gmail.com"' \
  --message-callback '
lines = [l for l in message.split(b"\n")
         if b"claude" not in l.lower() and b"co-authored-by" not in l.lower()]
return b"\n".join(lines)'

# (optional) squash main into a handful of logical commits
git checkout main
git rebase -i --root        # mark groups as "squash"/"fixup" in the editor

# filter-repo drops the remote for safety — re-add and force-push
git remote add origin https://github.com/shabazahmadme1-eng/runway_screening.git
git push -f origin main
```

## 2. Make main the default and delete the leftover branches

```bash
gh repo edit shabazahmadme1-eng/runway_screening --default-branch main
git push origin --delete local/nano-bootstrap
git push origin --delete claude/focused-ramanujan-pulfzw
```

(Or in the GitHub UI: Settings → Branches → set default to `main`, then the
Branches page → delete `local/nano-bootstrap` and `claude/…`.)

After this, the repo is a single clean `main` with no "Claude" anywhere.

# GitHub release checklist

## Required before public release

- [ ] Choose a software license and add `LICENSE` at the repository root. MIT is the recommended permissive option, subject to author approval.
- [ ] Replace every `GITHUB_REPOSITORY_URL` placeholder with the public GitHub URL.
- [ ] Create the public repository and upload the **contents** of `CAST-FV-GitHub/` as its root.
- [ ] Run `pip install -e ".[test]"` in a clean Python 3.10+ environment.
- [ ] Run `pytest -q` and confirm all tests pass.
- [ ] Run the CPU quick start and confirm that `outputs/steady/retained_state.png` is created.
- [ ] Confirm that no CFD reference arrays, target trajectories, checkpoints, out-of-scope solver code, credentials, or private absolute paths are present.
- [ ] Confirm that the three files in `assets/` come from the first manuscript and that the authors hold the right to publish them.
- [ ] Check the author name and contact details in `CITATION.cff`.
- [ ] Create a tagged GitHub release, recommended tag: `v0.1.0`.

## Recommended after public release

- [ ] Connect the GitHub release to Zenodo and reserve/archive a software DOI.
- [ ] Add the GitHub URL and, if available, software DOI to `CITATION.cff`.
- [ ] Replace the manuscript Code Availability statement with the text in `CODE_AVAILABILITY.md`.
- [ ] Rebuild both English and Chinese manuscripts and verify the final statement.
- [ ] Preserve the exact release used for the submitted manuscript.

## Public-scope audit

The release should contain algorithm code, tests, minimal examples, documentation, and selected conceptual manuscript figures only. It should not contain paper result data or implementation outside the present method.

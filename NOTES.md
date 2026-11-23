# NOTES

Third-party verification is not done yet for the full pipeline (the demo needs none of it). Pending for each repo before it goes under third_party/: install steps, license, checkpoint terms.

| repo | status |
|---|---|
| mapillary-python-sdk | to verify; token needed; images CC-BY-SA |
| VGGT | to verify; original checkpoint is non-commercial per the plan |
| gsplat | Apache-2.0 per the plan; needs CUDA |
| Depth-Anything-V2 | to verify |
| Grounded-SAM-2 | to verify |
| SegFormer / Mask2Former (transformers) | to verify |
| Trail datasets (IEEE DataPort, TOMD) | licenses and links to verify |

Demo build decisions:
- Scene parameter s equals x for the synthetic path, so validity compares risk at the recovered centerline's x against injected hazard spans.
- Hazard and class predictions are simulated per view; independent per-Gaussian noise makes k-view verification look better than real detector errors would.
- The off-corridor splat is dimmed in the heat recolor for readability.
- No git repo was initialized here; the enclosing Desktop repo owns this folder. .pre-commit-config.yaml is ready to use once this becomes its own repo.
- The web danger map (web/index.html) scores danger from elevation data (grade, side drop-off, ruggedness). This deliberately departs from the plan's rule that no DEM data may feed the score; the CV pipeline keeps that rule.
- Search by name queries Overpass live and was not exercised end to end because the public Overpass servers were overloaded while testing; the bundled trails were fetched earlier and are loaded locally.
- Bundled trail data covers 5 parks (snake den and blackstone returned few named ways; goddard returned none).

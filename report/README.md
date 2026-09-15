# Historical report — not regenerated research results

The PDFs, LaTeX, figures, and legacy result files describe the original four-label experiment. They predate the corrected pipeline and should not be presented as evidence for its performance.

Known issues: LaMini labels conflate base ancestry with teaching; original split sampling allows overlap; long prompts can remove response tokens; QuaRel is named in the text but absent from saved results; set-level results use only 80 sampled sets. See `docs/refactor.md` and the publication review for details.

`legacy_configs/` preserves the old configuration files for interpreting historical artifacts. It is not an executable experiment preset. The supplied `ta_report_final.pdf` differs from the tracked LaTeX draft; obtain its authoritative source before revising or rebuilding that final PDF. No numerical claims have been updated without rerunning experiments.

To compile the historical draft, if needed:

```bash
latexmk -pdf teacher_attribution_acl_draft.tex
```

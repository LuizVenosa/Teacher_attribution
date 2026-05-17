# ACL Report Draft

This folder contains the first ACL-style draft of the final project report.

Files:

- `teacher_attribution_acl_draft.tex`: main report source.
- `references.bib`: bibliography.
- `figures/01_training_loss_accuracy.png`: training loss and validation-accuracy curve.
- `figures/11_single_vs_set_level_comparison.png`: single-prompt versus set-level evaluation plot.

To compile with the official ACL style files:

1. Download the ACL style files from <https://github.com/acl-org/acl-style-files>.
2. Copy at least these files into this folder:
   - `acl.sty`
   - `acl_natbib.bst`
3. Compile:

```bash
pdflatex teacher_attribution_acl_draft
bibtex teacher_attribution_acl_draft
pdflatex teacher_attribution_acl_draft
pdflatex teacher_attribution_acl_draft
```

If a minimal TinyTeX install is missing ACL dependencies, install the missing LaTeX packages first, for example:

```bash
tlmgr install caption lineno
```

Before final submission:

- Add one confusion matrix figure or move it to an appendix.
- Check that the main body is at most 4 ACL pages. The assignment says limitations and references are outside the page count.

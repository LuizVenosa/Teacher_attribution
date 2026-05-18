# ACL Report

This folder contains the ACL-style final project report source, bibliography,
style files, figures, and compiled PDF.

Files:

- `teacher_attribution_acl_draft.tex`: main report source.
- `teacher_attribution_acl_draft.pdf`: compiled report PDF.
- `references.bib`: bibliography.
- `figures/00_latent_space_hero.png`: first-page latent-space visualization.
- `figures/01_training_loss_accuracy.png`: training loss and validation-accuracy curve.
- `figures/13_confusion_progression.png`: set-level confusion-matrix progression.

The ACL style files used for compilation are included:

```text
acl.sty
acl_natbib.bst
```

Compile with:

```bash
latexmk -pdf teacher_attribution_acl_draft.tex
```

If a minimal TinyTeX install is missing ACL dependencies, install the missing LaTeX packages first, for example:

```bash
tlmgr install caption microtype upquote
```

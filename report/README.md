# ACL Report Draft

This folder contains the first ACL-style draft of the final project report.

Files:

- `teacher_attribution_acl_draft.tex`: main report source.
- `references.bib`: bibliography.

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

Before final submission:

- Add the set-level accuracy plot if there is space.
- Add one confusion matrix figure or move it to an appendix.
- Check that the main body is at most 4 ACL pages. The assignment says limitations and references are outside the page count.

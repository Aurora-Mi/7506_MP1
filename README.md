# Project 1 submission

This directory is the clean submission package for DASE7506 MP1.

## Final reported result

- Validation BPB: **1.4598314480072423**
- Full-test BPB: **1.476789395067789**
- Checkpoint SHA256: `c8c30df36e1e0b29beb11c36155d3f36020d2f8ad8462ee55d89e25a77db574f`
- Evaluation protocol: `7506-mp1-wt2-v2`, CPU FP32, four threads

The frozen checkpoint and its matching source snapshots are under
`code/runs/final-frozen-wd06-refined-bigram/`. The English Word report is
`code/report/MP1_Report.pdf`.

## Directory structure

- `GUIDE.md`: coursework requirements.
- `code/`: source, fixed data/tokenizer, configurations, tests and instructions.
- `code/runs/final-frozen-wd06-refined-bigram/`: the only submitted model bundle.
- `code/report/MP1_Report.pdf`: final nine-page English report.
- `evidence/`: compact experiment summaries and recorded cost inventory; no old checkpoints.

No virtual environment, Python cache, smoke checkpoint, superseded checkpoint, or
temporary report artifact is included.

Run all commands from `code/`; see `code/README.md` for installation, verification,
training and evaluation instructions.

# Legacy CNN reference result

These files preserve the earlier CNN result from the previous repository state.

At epoch 7, the run had validation BCE 0.357594 and validation accuracy 0.860870. The earlier test evaluation reported 82.99% accuracy, which is a 17.01% 0/1 error.

This result predates the new CNN-vs-MLP protocol. It is retained only for provenance. It must not be compared against a newly trained MLP to select the final method. Run `run_stage2.py` so that both methods are trained and selected under one common protocol before the final test evaluation.

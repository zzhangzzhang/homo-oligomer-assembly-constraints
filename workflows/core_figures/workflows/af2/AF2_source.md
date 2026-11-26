# AlphaFold2 intervention implementation 

- ColabDesign `1.2.0b3`
- repository: `https://github.com/sokrypton/ColabDesign.git`
- requested revision: `gamma`
- commit: `3102efae40fb4c1dbe99812c80d8bf43251747df`
- Python 3.10.19
- JAX/JAXLIB 0.6.2

The installed ColabDesign commit implements:
- query one-hot versus position-specific MSA-profile conditioning;
- `cov_mask` propagation into extra-MSA and main Evoformer stacks;
- replacement of sequence-paired outer products with independently averaged
  outer products where the mask is zero.

## Other implementation

`scripts/inference/_af2_multimer_common.py` is the implementation of:

- QID filtering and MSA loading;
- query-bias selection;
- four covariance-mask strategies;
- model/seed/recycle settings;
- prediction/output handling.




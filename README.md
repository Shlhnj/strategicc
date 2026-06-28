# strategicc
STRATEGICC: State and Transition Integrated Economic-Environmental Accounting

A python package implementation of State-and-Transition Simulation Model (STSM) framework laid by Daniel et al (2016) (https://doi.org/10.1111/2041-210X.12597), integrated with the System of Economic-Environmental Accounting - Ecosystem Accounting (SEEA-EA) by United Nations (https://seea.un.org/ecosystem-accounting).

## Usage
###install the package


### Set Configuration

```
import stsm.config as cfg
from pathlib import Path

cfg.LULC_PATH          = Path("2022.tif")
cfg.STATE_CLASSES_CSV  = Path("25062026 State Class.csv")
cfg.TRANSITIONS_CSV    = Path("23062026_stsm_transition_probabilities.csv")
cfg.SPATIAL_MULT_CSV   = Path("27062026 Transition Spatial Multipliers.csv")
cfg.TRANSITION_MULT_CSV= Path("27062026 Transition Multipliers.csv")
cfg.MULT_DIR           = Path("mult_spat/")
cfg.OUT_DIR            = Path("stsm_tes_output_2_adjacency/")


cfg.ADJACENCY_STRENGTH   = 2
cfg.START_YEAR           = 2022
cfg.N_TIMESTEPS          = 30
cfg.N_ITERATIONS         = 100    # ← change this to run more/fewer iterations
cfg.RNG_SEED             = 42

cfg.USE_ADJACENCY        = True
cfg.USE_SPATIAL_MULT     = True
cfg.USE_TRANS_MULTIPLIER = True
```

### Diagnose Configuration

```
from stsm import STSMEngine

engine = STSMEngine.from_config()
engine.load()
engine.diagnostic()
```

### Run Engine

```
engine.run()
```

### Show Summary Plot

```
from stsm import outputs

summary_dir = engine.out_dir / "summary"

print("Building summary tables...")
area_df, trans_df = outputs.build_summary_tables(engine.iter_dirs, summary_dir)

print("Plotting area envelope...")
outputs.plot_area_envelope(area_df, engine.classes, summary_dir)

print("Plotting transition envelope...")
outputs.plot_transition_envelope(trans_df, summary_dir)

Show plots inline
from IPython.display import Image, display
display(Image(str(summary_dir / "area_envelope.png")))
display(Image(str(summary_dir / "transition_envelope.png")))
```

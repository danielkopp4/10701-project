# 10701 Project

To download the data
```bash
python -m src.get_data
```

To run experiments
```bash
python -m src.run_experiments
```

To run analysis of experiments
```bash
NOT YET DONE
```

TODO:
- [ ] generate analysis given experiment results
- [ ] add more models 
- [ ] add more experiments
- [ ] validate outcomes of the analysis
- [ ] figure out why BMI only errors out and gives NAN correlation


Experiments we should run
- BMI only + all models (baseline, standard for medicine)
- Waist to height ratio + all models (proposed alternative)
- BMI + latent common cause (recommended by masters et al) + all model types
- All available features (naive approach, will have better IID performance but worse causal performance)
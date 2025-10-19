
## run simulations
```bash
python run_multi.py N_SIMULATIONS=100000 SAVE_SNAPSHOTS=1 OUTPUT_DIR="/Volumes/LaCie/Thesis_work/Simulations_base" --BATCH_SIZE=250
```

## run plot endpoint and pcf images
### By parameters
```bash
patrik$ python plot_sim_overview.py --output-dir /Volumes/LaCie/Thesis_work/Simulations_base --by-params --D 0.01 --tol-log10-d 0.02 --radius 7.5 --tol-r 2.5 --log10-alpha -5.1 --tol-log10-alpha 0.2
```

### By sim\_id
```bash
python plot_sim_overview.py --output-dir /Volumes/LaCie/Thesis_work/Simulations_base --sim-id 500
```

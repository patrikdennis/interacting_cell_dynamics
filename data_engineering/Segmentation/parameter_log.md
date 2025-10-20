A1: 
```bash
python  Franzeg.py --folder "/Volumes/LaCie/Thesis_work/Test_data"                    --out-parent "/Volumes/LaCie/Thesis_work"                    --out-folder-name "A1_segmented_test_data" --offset 1.0 --sigma 1.2 --contrast-min 0.015 --grad-low 0.016 --grad-high 0.040 --grad-mean-min 0.025
```

Below works even better!
```bash
python  Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/test_data_A1"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A01_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```

A10: 
```bash
python  Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/test_data_a10"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A10_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```

A11:
```bash
python  Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/Test_data"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A1_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```

A12:
```bash
python Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/test_data_A12"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A12_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```



Working well with both A1, A11 and A10


## trackpy feature extraction

A1:

For testing:
```bash
python Franzeg_optimized_v2.py   --folder "/Users/patrik/cell_diffusion_modelling/558/A1_transformed_subset"   --out-parent "/Users/patrik/cell_diffusion_modelling/558"   --out-folder-name "A1_level_subsubsubsettest_segmented"   --sigma 1.0 --offset 0.8 --levels 12   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15 --mode all --square-size 5 --approximation-level 0.005
```

### Final:
```bash
python Franzeg_optimized.py   --folder "/Users/patrik/cell_diffusion_modelling/558/A1_transformed_subset"   --out-parent "/Users/patrik/cell_diffusion_modelling/558"   --out-folder-name "A1_level_subsubsubsettest_segmented"   --sigma 1.0 --offset 0.8 --levels 12   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15 --mode all --square-size 5 
```


## run trackpy

A1:

```bash
python run_tracking.py --input-folder /Volumes/LaCie/Thesis_work/558/A1_processed/trackpy_features --output-folder /Volumes/LaCie/Thesis_work/558/A1_processed/track_results --diameter 5 --minmass 100 --max-displacement 20 --memory 4 --min-length 4
```

## extract trajectory data

```bash
python trajectory_analysis.py   --infile /Volumes/LaCie/Thesis_work/558/A1_processed/track_results/trajectories.csv --features_folder /Volumes/LaCie/Thesis_work/558/A1_processed/trackpy_features --output_folder /Volumes/LaCie/Thesis_work/558/A1_processed/analysis_plots   --pixelsize 1.0   --unit-name "um"   --time-base "seconds" --plots all
```

### updated

```bash
python trajectory_analysis.py   --infile /Volumes/LaCie/Thesis_work/558/A10_processed/track_results/trajectories.csv --features_folder /Volumes/LaCie/Thesis_work/558/A10_processed/trackpy_features --output_folder /Volumes/LaCie/Thesis_work/558/A10_processed/analysis_plots   --pixelsize 1.0   --unit-name "um"   --time-base "seconds" --plots all --msd_fit_points 15
```



## feature extraction fogbank

```bash
python Fogbank_features.py --folder /Users/patrik/cell_diffusion_modelling/558/new_fogbank_labelled --out-dir /Users/patrik/cell_diffusion_modelling/558/new_fogbank_labelled --method edt_weighted --alpha 2.0 --square-size 5 --workers 4
```
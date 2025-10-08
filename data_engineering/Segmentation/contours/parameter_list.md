A1: 
```
python  Franzeg.py --folder "/Volumes/LaCie/Thesis_work/Test_data"                    --out-parent "/Volumes/LaCie/Thesis_work"                    --out-folder-name "A1_segmented_test_data" --offset 1.0 --sigma 1.2 --contrast-min 0.015 --grad-low 0.016 --grad-high 0.040 --grad-mean-min 0.025
```

Below works even better!
```
python  Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/test_data_A1"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A01_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```

A10: 
```
python  Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/test_data_a10"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A10_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```

A11:
```
python  Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/Test_data"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A1_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```

A12:
```
python Franzeg.py   --folder "/Volumes/LaCie/Thesis_work/test_data_A12"   --out-parent "/Volumes/LaCie/Thesis_work"   --out-folder-name "A12_segmented_test_data"   --sigma 1.0 --offset 0.8 --levels 30   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15
```



Working well with both A1, A11 and A10


## trackpy feature extraction

A1:

```
python Franzeg_optimized_v2.py   --folder "/Users/patrik/cell_diffusion_modelling/558/A1_transformed_subset"   --out-parent "/Users/patrik/cell_diffusion_modelling/558"   --out-folder-name "A1_level_subsubsubsettest_segmented"   --sigma 1.0 --offset 0.8 --levels 12   --grad-low 0.010 --grad-high 0.030 --grad-mean-min 0.020   --contrast-min 0.010   --inside-max 0.70 --inside-percentile 0.60   --dark-rel-min 0.015 --dark-rel-each 0.015 --min-dark-edge-cover 0.45   --min-length 25 --bg-sigma 15 --mode all --square-size 5 --approximation-level 0.005
```
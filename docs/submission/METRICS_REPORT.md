# Anveshak Round 1 Metrics Report

Prepared on 27 November 2026 for the Techfest IIT Bombay open world remote sensing detection challenge.

## 1. Summary

I trained one detector from random initialisation and improved it in stages. My submitted model is **B_final3**, which descends from a single lineage A, A2, A3, A4 and then B_final3.

| Measure | Value |
|---|---|
| Known class mAP at IoU 0.5 on Set A validation (mAP_K) | **0.394** |
| Unknown F1 on my 430 object proxy | **0.122** |
| Estimated overall score (100 times the harmonic mean) | **about 18.6** |
| Submitted file | r1_final3_g025.csv with 780,021 rows and 944 unknown rows |
| Official validator | passes with no errors |
| Reproduction from the checkpoint | 780,021 of 780,021 rows match exactly |

The overall score is only an estimate because the test labels are hidden. I computed it from my own held out data as explained in section 6.

## 2. Data I used

I used only the data released for Round 1. I used no external data and no pretrained weights.

| Split | Tiles | Use |
|---|---|---|
| Set A train | 8,707 | base training for classes 0 to 29 |
| Set A validation | 952 (80,218 boxes) | held out the whole time and used for every known class score |
| Set B | 677 (505 train and 172 validation) | few shot classes 30 to 39 |
| Test | 1,636 | inference only |

The test images were never used for training, tuning or labelling.

## 3. Hardware and software

| Item | Detail |
|---|---|
| GPU | NVIDIA RTX 5050 Laptop with 8 GB of memory |
| System | Lenovo LOQ, Windows 11, 16 GB RAM |
| Framework | Ultralytics 8.4.215 with PyTorch 2.11 and CUDA 12.8 |
| Architecture | YOLO11 at medium scale with an extra high resolution P2 detection head |
| Output slots | 52 in total, namely 40 known classes, 10 reserved for Round 2, one unknown slot and one objectness slot |

## 4. Results of the main lineage

All mAP_K numbers below are AP at IoU 0.5 averaged over the 30 Set A classes on the held out Set A validation split. I scored every model under the same post processing so that the numbers are comparable.

| Stage | What changed | Epochs | GPU hours | mAP_K |
|---|---|---|---|---|
| A | trained from scratch at 1024 pixels, batch 2 | 250 | 14.2 | 0.295 |
| A2 | continued on 640 pixel tiles at batch 8 with milder zoom | 12 | 6.3 | 0.350 |
| A3 | 12 more epochs with the same recipe | 12 | 6.1 | 0.379 |
| A4 | 12 more epochs with the same recipe | 12 | 6.5 | 0.404 |
| A5 | 12 more epochs with the same recipe | 12 | 6.1 | 0.418 |

The tile recipe gave the largest gains of the project. Each further block of 12 epochs still helped, though the gain shrank from 0.055 to 0.015.

## 5. Results of the Set B fine tunes

Each fine tune starts from a base model and adds the ten few shot classes. I measured how much Set A accuracy each one kept.

| Fine tune | Base | Recipe | Epochs | GPU hours | mAP_K | Loss against its base |
|---|---|---|---|---|---|---|
| B | A | 1024 pixels, Set B pseudo labels | 60 | 9.5 | 0.281 | 0.014 |
| B2 | A | as B plus copy paste | 60 | 8.2 | 0.284 | 0.011 |
| B_final | A2 | 1024 pixels with 6,000 Set A replay tiles | 50 | 14.5 | 0.323 | 0.027 |
| B_final2 | A3 | 640 pixel tiles at half learning rate | 25 | 10.2 | 0.368 | 0.011 |
| **B_final3** | **A4** | **640 pixel tiles at half learning rate** | **25** | **9.6** | **0.394** | **0.009** |
| B_final4 | A5 | 640 pixel tiles | 25 | 9.8 | 0.414 | 0.005 |

Matching the fine tune recipe to the base recipe cut the forgetting from 0.027 to about 0.01.

Few shot accuracy on the Set B validation split was 0.079 for B, 0.075 for B2 and 0.092 for B2 with test time augmentation. The later fine tunes trained on all of Set B including its validation tiles, so I do not report a Set B score for them.

## 6. Unknown detection

The competition counts unknown boxes without ranking, so every false unknown lowers precision. I built a proxy because the training data has no unknown labels. My Set A models never saw any Set B tile, so the ten Set B classes behave like unknown objects for them. On all of Set B that gives about 430 unknown objects.

### 6.1 Unknown F1 by base model

| Base | Original penalty (gamma 1) | Softer penalty (gamma 0.25) |
|---|---|---|
| A3 | 0.106 | 0.119 |
| A4 | 0.090 | **0.122** |
| A5 | 0.090 | 0.095 |

Stronger base models became better at known classes but more confident that unfamiliar objects belong to a known class. The softer penalty recovered most of that loss for A4.

### 6.2 Tuning the gamma setting on A3

| Gamma | Unknown F1 |
|---|---|
| 1 | 0.106 |
| 0.5 | 0.112 |
| 0.25 | **0.119** |
| 0.1 | 0.111 |
| 0 | 0.030 |

The curve is smooth with a clear peak at 0.25. The cost on known classes was negligible because mAP_K moved from 0.3679 to 0.3676.

### 6.3 Estimated overall score

| Candidate | mAP_K | Unknown F1 | Estimated score |
|---|---|---|---|
| r1_final2_g025 (A3 base) | 0.368 | 0.119 | 18.0 |
| **r1_final3_g025 (A4 base)** | **0.394** | **0.122** | **18.6** |
| r1_final4 at gamma 0.25 (A5 base) | 0.414 | 0.095 | 15.5 |

## 7. Experiments that shaped the recipe

### 7.1 Recipe ablation

I fine tuned Stage A for 10 epochs on a quarter of the data under four settings.

| Setting | mAP_K |
|---|---|
| Control at 1024 pixels, batch 2 | 0.295 |
| 640 pixel tiles, batch 8 | 0.302 |
| 640 pixel tiles, batch 8, zoom 0.25 | **0.309** |
| as above plus stronger class rebalancing | 0.307 |

The tile recipe won and stronger rebalancing did not help, so A2 used the tile recipe with the normal sampling.

### 7.2 The unknown slot was hiding known objects

A per class confusion check on Stage A showed that 3,606 known objects were output as unknown and then removed by the conflict rule. Seven classes lost between 30 and 62 percent of their objects this way. Adding a minimum size of 48 pixels and a cap of 10 unknowns per tile fixed most of it. For example class e0l0hksz3i4 rose from 0.05 to 0.25 and class tmxh1pmfl4h rose from 0.006 to 0.14 with no retraining.

### 7.3 Test time augmentation

Four flipped views merged by the same suppression step raised B2 by 0.014 on Set A and by 0.017 on the Set B few shot classes, so every final CSV uses it.

## 8. Per class results of the submitted model

AP at IoU 0.5 on Set A validation for B_final3, next to its base A4 and the original Stage A.

| Class | Boxes | Stage A | A4 | B_final3 |
|---|---|---|---|---|
| uap2j6grlou | 43,179 | 0.700 | 0.731 | 0.720 |
| o1f7fks8if4 | 28,737 | 0.783 | 0.822 | 0.810 |
| e0l0hksz3i4 | 1,874 | 0.052 | 0.353 | 0.350 |
| qc8qjxp16ky | 960 | 0.242 | 0.525 | 0.512 |
| tmxh1pmfl4h | 775 | 0.006 | 0.262 | 0.252 |
| 62q5gbazvqv | 630 | 0.296 | 0.547 | 0.512 |
| 08pke08qywd | 578 | 0.126 | 0.235 | 0.215 |
| bb86sjpmpbo | 531 | 0.221 | 0.470 | 0.487 |
| j6yvz6udoms | 529 | 0.006 | 0.069 | 0.059 |
| mk6ikfy1yt2 | 249 | 0.739 | 0.854 | 0.861 |
| sexxkfu6417 | 212 | 0.229 | 0.396 | 0.352 |
| jy6hz4dfl2n | 202 | 0.019 | 0.186 | 0.157 |
| jawayrw5297 | 169 | 0.060 | 0.380 | 0.365 |
| z7nu4z33dso | 167 | 0.105 | 0.193 | 0.205 |
| e6t3x2t65v4 | 157 | 0.030 | 0.153 | 0.160 |
| 9low6xa8q8c | 156 | 0.596 | 0.666 | 0.651 |
| p55s511cysh | 148 | 0.033 | 0.150 | 0.147 |
| zgybjlga9ti | 116 | 0.379 | 0.557 | 0.528 |
| 069wa24mde3 | 112 | 0.012 | 0.075 | 0.072 |
| ci914dy70uq | 110 | 0.186 | 0.276 | 0.283 |
| ffx0247nfiw | 86 | 0.016 | 0.209 | 0.185 |
| 96riulz0zvo | 84 | 0.309 | 0.466 | 0.405 |
| 5ybyfcxlqmt | 76 | 0.455 | 0.633 | 0.682 |
| 4sptj80bvcz | 73 | 0.014 | 0.119 | 0.152 |
| y9apdit2y41 | 73 | 0.009 | 0.016 | 0.026 |
| g53ktwtqo6v | 66 | 0.907 | 0.861 | 0.851 |
| o4crg3ag82w | 60 | 0.142 | 0.613 | 0.574 |
| skemv4dvxc7 | 55 | 0.329 | 0.521 | 0.469 |
| xvoincbmb42 | 39 | 0.007 | 0.049 | 0.055 |
| jww9d410omp | 15 | 0.486 | 0.729 | 0.740 |
| **Mean** | | **0.250** | **0.404** | **0.395** |

The Stage A column uses its original post processing and the B_final3 column uses the setting from 26 November. The submitted file uses gamma 0.25, which scores 0.394.

## 9. Known weaknesses

* A few classes are still confused with the two dominant classes, for example j6yvz6udoms and p55s511cysh with o1f7fks8if4, and xvoincbmb42 with uap2j6grlou. Stronger sampling did not fix this.
* Class y9apdit2y41 has large objects of about 120 pixels and is still mostly missed.
* Unknown F1 is the weaker half of the score and depends on a proxy of about 430 objects, so I expect some variation on the test set.

## 10. Compute

The submitted lineage used about 43 GPU hours. All stages, ablations and checks together used about 110 GPU hours on the single laptop GPU.

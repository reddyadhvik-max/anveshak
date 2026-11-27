# Anveshak Round 1 Submission

Techfest IIT Bombay open world remote sensing detection challenge, submitted November 2026.

## 1. What I set out to build

I built one object detector for satellite and aerial tiles. It finds 30 base classes, learns 10 more classes from only a few examples, and marks anything it does not recognise as unknown. I trained it entirely from random initialisation on the data the organisers released, and I kept it as a single model lineage so that Round 2 can continue from the same checkpoint.

My final submission is **r1_final3_g025.csv**, produced by the checkpoint **runs/B_final3/weights/last.pt** together with **configs/postproc.yaml**.

## 2. Rules I followed

* I used no pretrained weights of any kind. Every model starts from my own configuration with random weights, and I disabled the Ultralytics check that silently downloads a small pretrained model.
* I used no external data. I trained only on Set A and Set B.
* I used the test images for inference only. I never labelled them, never trained on them and never tuned on them.
* I used no open vocabulary detector. Unknown detection comes from my own logic inside a closed set detector.
* I kept one lineage, A then A2, A3, A4 and finally B_final3. I did not merge outputs from different models.
* I checked that rerunning the checkpoint reproduces my CSV. All 780,021 rows matched exactly.

## 3. Data

Set A gave me 8,707 training tiles and a validation split of 952 tiles with 80,218 boxes over the 30 base classes. I held that validation split out for the whole project and used it for every known class score.

Set B gave me 677 tiles containing the 10 few shot classes, with between 15 and 79 examples per class. Set B tiles also contain many unlabelled base class objects, which shaped how I used them.

The data is very imbalanced. Two classes account for about 72,000 of the 80,218 validation boxes, and the median object is only about 24 pixels across. To give rare classes more exposure I used repeat factor sampling, which raised each epoch from 8,707 to 11,347 samples.

## 4. Model

I used YOLO11 at medium scale from Ultralytics 8.4.215 with PyTorch 2.11. Because the objects are so small, I added an extra P2 detection head that works at a quarter of the input resolution.

I fixed the output layer at 52 slots from the very first training run.

* Slots 0 to 29 hold the base classes.
* Slots 30 to 39 hold the few shot classes.
* Slots 40 to 49 are reserved for the Round 2 classes, so the network never needs to be rebuilt.
* Slot 50 is a dedicated unknown class and slot 51 is a general objectness score.

## 5. How I trained it

I trained everything on one laptop GPU, an NVIDIA RTX 5050 with 8 GB of memory.

### 5.1 Making training fit on the laptop

The first benchmarks were very slow because Windows quietly spilled GPU memory into system RAM. I turned off the driver setting responsible and changed the box assignment step to process very dense tiles one image at a time and in chunks. That made training about 2.5 times faster and removed the memory crashes. I also built a small local control page with pause, stop and resume buttons, and a pipeline runner that executes every step in order and resumes after interruptions. That design let me recover in minutes when a graphics driver crash interrupted training on 26 November.

### 5.2 Stage A, the base detector

I trained Stage A for 250 epochs at 1024 pixels with batch 2 and gradient accumulation to an effective batch of 64. It reached 0.295 mAP_K on Set A validation.

When I studied the learning curves I found the model was underfitting, because the training loss stayed above the validation loss and both stopped improving around epoch 100. I ran a controlled ablation to find out why. Training on native 640 pixel windows cut from each tile, with batch 8 and a milder zoom augmentation, beat the original recipe. Stronger class rebalancing did not help.

### 5.3 Continuing the base model with the better recipe

I continued from Stage A in blocks of 12 epochs with the tile recipe.

| Model | mAP_K on Set A validation |
|---|---|
| Stage A | 0.295 |
| A2 | 0.350 |
| A3 | 0.379 |
| A4 | 0.404 |
| A5 | 0.418 |

This was the biggest improvement of the project. Many weak classes rose three to ten times, for example o4crg3ag82w from 0.14 to 0.61.

## 6. How I learned the few shot classes

### 6.1 Pseudo labels for Set B

Set B tiles only label the few shot classes, so a model trained on them directly would learn that base class objects are background. To prevent that, I ran my base model over Set B. Confident base class detections scoring at least 0.4 became labels. Uncertain detections between 0.1 and 0.4 that touched no label were greyed out, so the model was neither rewarded nor punished for them. For B_final3 this gave 65,617 pseudo labels.

### 6.2 Replay to avoid forgetting

I mixed every Set B tile six times with 6,000 Set A tiles, so the model kept practising the base classes while learning the new ones.

### 6.3 Copy paste

I cut the few shot objects out of Set B and pasted 2,864 of them into 1,500 Set A tiles to give the new classes more varied backgrounds. I inspected samples by eye. I also tried feathered edges but saw little benefit, so I kept the simple version.

### 6.4 Matching the recipe

My first full fine tune at 1024 pixels lost 0.027 mAP_K on Set A because it no longer matched how the base had been trained. When I fine tuned with the same tile recipe as the base and halved the learning rate, the loss fell to about 0.01. B_final3 kept 0.394 out of its base model's 0.404 while adding the ten new classes.

## 7. How I detect unknown objects

### 7.1 The unknown score

For every candidate box I compute an unknown score as the larger of two signals. The first is the dedicated unknown slot. The second is the objectness score multiplied by a penalty that grows as the best known class score grows. I raised that penalty to the power gamma, and I found that gamma 0.25 works best, because the penalty no longer overwhelms objects that only loosely resemble a known class.

### 7.2 A proxy for measuring unknowns

The training data contains no labelled unknown objects, so I built a proxy. My Set A models never saw any Set B tile, so for them the ten Set B classes are genuinely unfamiliar. Scoring unknown boxes against those objects on all of Set B gave me about 430 test objects for tuning. I always tuned on this proxy with the base model and never on the test set.

### 7.3 What the proxy taught me

* The unknown slot was stealing known objects. 3,606 small known objects were output as unknown and then deleted by the conflict rule. A minimum size of 48 pixels and a limit of 10 unknowns per tile fixed most of it without retraining.
* Stronger base models are worse at flagging unfamiliar objects, because they assign them confidently to a known class. Unknown F1 fell from 0.106 on A3 to 0.090 on A4 under the original penalty.
* The softer gamma 0.25 penalty lifted A4 to 0.122 at essentially no cost to known classes. That is why I chose the A4 lineage for the final submission over the stronger but less flexible A5 lineage.

## 8. Post processing

* Known boxes are kept down to a score of 0.001 because mAP rewards extra low score boxes, with class wise suppression at IoU 0.6 and at most 900 per tile.
* Unknown boxes need a score of at least 0.10 and a size of at least 48 pixels, with suppression at IoU 0.5 and at most 10 per tile.
* I apply the competition conflict rule myself before writing the file, so where a known box and an unknown box overlap at IoU 0.5 or more only the higher score remains.
* I use test time augmentation with four flipped views of every tile, merged by the same suppression step. It added about 0.014 mAP_K.
* Every file stays within 1,000 rows per tile and passes the official validate_predictions.py script.

## 9. Final results

| Measure | Value |
|---|---|
| Known class mAP_K on Set A validation | 0.394 |
| Unknown F1 on my 430 object proxy | 0.122 |
| Estimated overall score | about 18.6 |
| Rows in the submitted file | 780,021 including 944 unknown boxes |
| Reproduction check | 0 mismatched rows |

I chose this file by estimated overall score rather than known class accuracy. B_final4 reached a higher 0.414 mAP_K, but its weaker unknown detection gave a lower estimated score of 15.5.

## 10. What I would do next

* Address the classes that are still confused with the two dominant classes, possibly with targeted hard negative examples.
* Improve detection of very large objects such as class y9apdit2y41.
* Develop a stronger unknown signal, since unknown F1 is now the main limit on the overall score.

## 11. Tools I used

* Ultralytics YOLO11 and PyTorch for the detector and training.
* Python with OpenCV and NumPy for data preparation, pseudo labelling, copy paste and scoring.
* The official validator script supplied by the organisers.

## 12. Files in this submission

| File | sha256 |
|---|---|
| runs/B_final3/weights/last.pt | bd13436cb324ce0b07fd49183601f2a8ebb14024f7abbb0eb0bef6d4668f9b84 |
| configs/postproc.yaml | 478cabd86fba2129d6728b08575c8c7a49a38bf29f8d329b1ace77a28a53c245 |
| r1_final3_g025.csv | 7366366eae7604d735d93a24687d08f35ca6fe205ae66541d3ccafb111e2cd65 |

# Section Three — ten seeds, sum and mean

360 epochs per fit; fixed .5 test threshold; shared clean partition.

| Seed | DDESONN FALSE: sum | DDESONN TRUE: mean | Keras mean | Keras sum |
|---|---:|---:|---:|---:|
| 1 | 90.90% | 81.30% | 80.42% | 87.78% |
| 2 | 92.27% | 81.17% | 92.14% | 89.90% |
| 3 | 87.78% | 79.05% | 85.04% | 84.91% |
| 4 | 86.16% | 77.93% | 88.03% | 87.41% |
| 5 | 94.51% | 80.92% | 91.02% | 89.78% |
| 6 | 94.14% | 82.67% | 89.78% | 92.39% |
| 7 | 94.01% | 83.17% | 88.65% | 96.38% |
| 8 | 90.15% | 77.56% | 84.66% | 84.66% |
| 9 | 91.27% | 81.55% | 90.02% | 91.40% |
| 10 | 91.52% | 81.55% | 89.53% | 87.53% |
| **Mean** | 91.27% | 80.69% | 87.93% | 89.21% |
| SD (percentage points) | 2.73 | 1.89 | 3.55 | 3.57 |


All 40 fits completed 360 epochs. Exact predictor overlap across splits is zero.
Keras retains batches of 16 and its final checkpoint; DDESONN uses all 3,500
training rows and its best validation checkpoint. Other archived framework
differences remain. These are ten training seeds on one clean partition.

# Automl comparison

This is code for comparison of automatic machine learning libraries:

 - [auto-sklearn](https://github.com/automl/auto-sklearn)
 - autoML from [h2o](https://www.h2o.ai)
 - [mljar](https://mljar.com)

### Datasets used for comparison

| Dataset Id | Name | Rows | Columns |
| - | - | - | - |
| 3 | kr-vs-kp | 3196 | 36 |
| 24 | mushroom | 8124 | 22 |
| 31 | credit-g | 1000 | 20 |
| 38 | sick | 3772 | 29 |
| 44 | spambase | 4601 | 57 |
| 179 | adult | 48842 | 14 |
| 715 | fri_c3_1000_25 | 1000 | 25 |
| 718 | fri_c4_1000_100 | 1000 | 100 |
| 720 | abalone | 4177 | 8 |
| 722 | pol | 15000 | 48 |
| 723 | fri_c4_1000_25 | 1000 | 25 |
| 727 | 2dplanes | 40768 | 10 |
| 728 | analcatdata_supreme | 4052 | 7 |
| 734 | ailerons | 13750 | 40 |
| 735 | cpu_small | 8192 | 12 |
| 737 | space_ga | 3107 | 6 |
| 740 | fri_c3_1000_10 | 1000 | 10 |
| 741 | rmftsa_sleepdata | 1024 | 2 |
| 819 | delta_elevators | 9517 | 6 |
| 821 | house_16H | 22784 | 16 |
| 822 | cal_housing | 20640 | 8 |
| 823 | houses | 20640 | 8 |
| 833 | bank32nh | 8192 | 32 |
| 837 | fri_c1_1000_50 | 1000 | 50 |
| 843 | house_8L | 22784 | 8 |
| 845 | fri_c0_1000_10 | 1000 | 10 |
| 846 | elevators | 16599 | 18 |
| 847 | wind | 6574 | 14 |

To download datasets you need to register on [openML](https://www.openml.org/) and set OPENML_KEY in your environment.

### Methodology

1. Each dataset was divided into train and test set (70%/30%).
2. The autoML package was trained on train set. There was 1 hour limit for training.
3. Final autoML model was used to compute predictions on test set (samples not used for training).
4. The [logloss](https://www.kaggle.com/wiki/LogLoss) was used to asses model performance (the lower the better).
5. The process was repeated 10 times (with different seeds), results are average over 10 repeats.

### Results

| Dataset Id | Auto-sklearn | H2O          | MLJAR        |
|---------|--------------|--------------|--------------|
| 179     | 0.4977899919 | 0.3152976149 | **0.3049036708** |
| 24      | 0.0008299585 | 0.0064581712 | **0.000003843**  |
| 3       | 0.1971600449 | 0.0259192684 | **0.0188968126** |
| 31      | 0.5083364979 | 0.5619242106 | **0.4939436971** |
| 38      | 0.1654747739 | 0.045179262  | **0.0389534345** |
| 44      | 0.3838450926 | 0.1360053817 | **0.125079404**  |
| 715     | 0.2818374134 | 0.236469178  | **0.2068510073** |
| 718     | 0.2638171184 | 0.2789765963 | **0.2419061124** |
| 720     | 0.4998641475 | 0.4516830437 | **0.4309945916** |
| 722     | 0.3908241383 | 0.0322915583 | **0.0298104695** |
| 723     | 0.3293591298 | 0.2739778619 | **0.2500752244** |
| 727     | 0.3978124619 | 0.1595664723 | **0.1495902549** |
| 728     | 0.0927113633 | 0.0331800086 | **0.0198811926** |
| 734     | 0.5432702419 | 0.2738200987 | **0.2548240677** |
| 735     | 0.4268106101 | 0.1811306173 | **0.1639352045** |
| 737     | 0.4824914304 | **0.330549118**  | 0.367133044  |
| 740     | 0.3142420934 | 0.2254379351 | **0.2149655921** |
| 741     | 0.5729526032 | 0.5671571325 | **0.5247308762** |
| 819     | 0.4847074932 | 0.2953733834 | **0.2865255395** |
| 821     | 0.5601634059 | 0.2545560683 | **0.241073802**  |
| 822     | 0.5179014444 | **0.2326348019** | 0.2373530782 |
| 823     | 0.4203668177 | 0.0664409033 | **0.0403094539** |
| 833     | 0.442981634  | 0.3944518914 | **0.3769150535** |
| 837     | 0.3427939126 | 0.2430881733 | **0.2064752939** |
| 843     | 0.4363679407 | 0.2670714458 | **0.2531006498** |
| 845     | 0.4559840276 | 0.3148631409 | **0.2550298912** |
| 846     | 0.5457531577 | 0.2467550186 | **0.2401925153** |
| 847     | 0.3860761231 | 0.3245902333 | **0.3049773203** |

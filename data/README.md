Put all data (both [raw competition data](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data) and [processed data](https://www.kaggle.com/datasets/dangnh0611/ecg-digitization-processed-data/)) into this directory. Restructure so that data directory looks like (outputs of `tree -L 2 ./`):
```
❯ tree -L 2 ./
./
|-- processed
|   |-- REFERENCE_TEMPLATE_FS100.png
|   |-- cv
|   |-- imc_results.json
|   |-- keypoints_by_homo.csv
|   |-- keypoints_by_homo.npy
|   |-- keypoints_by_model.csv
|   |-- keypoints_by_model.npy
|   |-- keypoints_by_round1.csv
|   |-- keypoints_by_round1.npy
|   |-- keypoints_by_round2.csv
|   |-- keypoints_by_round2.npy
|   |-- pseudo_label
|   |-- pseudo_test
|   |-- reference_keypoints.json
|   `-- templates
`-- raw
    |-- sample_submission.parquet
    |-- test
    |-- test.csv
    |-- train
    `-- train.csv

9 directories, 14 files
```


**Step-by-step details**:
- Download the [official competition data](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/data) and put to [./data/raw/](./data/raw)
- Download the processed data from [my Kaggle dataset](https://www.kaggle.com/datasets/dangnh0611/ecg-digitization-processed-data): More details and explainations about how each part/file was generated are listed under [../docs/DATA_PROCESSING.md](../docs/DATA_PROCESSING.md)

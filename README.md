## Project Organization

Here is the structure of the project. The main file to look at is main.ipynb. It was more convenient to use a notebook for running the code on the cluster.

```
├── README.md
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports
│   └── figures
│
├── main.ipynb         <- Main notebook with some data exploratory, train & tuning models and plots
|
├── results_cnn.json
├── results_ViT_diff_head_cls_model.json
├── results_ViT_diff_head_mlp_over_rep.json
├── results_Vit_diff_head_no_cls_model.json
├── results_ViT_ref_book.json
|
└── cancer_classifier   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes cancer_classifier a Python module
    ├── config.py               <- Store useful variables and configuration
    |
    ├── modeling                <- Contrains models, training and tuning code
    |   |
    |   ├── models              <- Contains models
    |   |   ├── cnnmodel.py
    |   |   ├── vit_diff_head_cls
    |   |   ├── vit_diff_head_mlp_over_rep.py
    |   |   ├── vit_diff_head_no_cls
    |   |   └──vit_ref_book.py
    |   |
    │   └── train.py            <- Code to train models and tune models
    |
    └── processing              <- Contrains data loader, data processing and plot code
        │
        ├── data_loader.py      <- The main data loader
        ├── dataset_loader.py   <- Not relevent anymore, it is still called in some old notebooks
        ├── image_utils.py      <- Utility functions for image processing
        └── plots.py            <- Code to create visualizations
```



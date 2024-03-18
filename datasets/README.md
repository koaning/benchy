I wanted to host a few datasets that can be found on [Kaggle](https://www.kaggle.com/competitions) to make it easier to run some benchmarks on these. An effort is made to respect the License of each dataset and as a rule we'll only consider datasets of past competitions in this repository.

## Playground Series 

Datasets with the [CC by 4.0](https://creativecommons.org/licenses/by/4.0/) licenses were copied into this project and turned into parquet. Only the `train.csv` files are kept. The orignal links to any of the files listed here can be derived from the dataset name. 

For example, the file on this path ...

```
datasets/playground-series-s3e22.parquet
```

... can be found on Kaggle via ... 

```
https://www.kaggle.com/competitions/playground-series-s3e22/data
```

... note how `s3e22` is the key identifier that's used in both places.

### One edit

We perform one, and only one, change to the original data which is what we remove the `id` column when you load from Python. This column should be avoided for modelling purposes anyway, so it felt safe to remove.

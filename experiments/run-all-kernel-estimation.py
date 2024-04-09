from subprocess import call
from data.datasets import DATASETS


num_splits = 10

for dataset in DATASETS.keys():
    for split_id in range(num_splits):

        call(
            [
                "python",
                "experiments/kernel-estimation.py",
                "--dataset",
                dataset,
                "--seed-dataset",
                "0",
                "--num-splits",
                str(num_splits),
                "--split-id",
                str(split_id),
                "--max-datapoints",
                "1024",
                "--lengthscale",
                "1.0",
                "--output-scale",
                "1.0",
                "--noise-std",
                "0.1",
                "--num-ensembles",
                "1",
                "--seed-training",
                str(split_id),
                "--num-steps",
                "5000",
                "--learning-rate",
                "0.01",
                "--num-trials",
                "1000",
                "--sampler-learning-rate",
                "0.001",
                "--sampler-num-steps",
                "1000",
            ]
        )

from subprocess import call
from data.datasets import DATASETS


num_splits = 5

for dataset, dataset_class in DATASETS.items():
    for split_id in range(num_splits):
        print(200*"=")
        print(dataset_class.name)
        print(200*"=")
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
                "256",
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
                "100",
                "--sampler-learning-rate",
                "0.001",
                "--sampler-num-steps",
                "2000",
                "--frame",
                "ortho"
            ]
        )
        

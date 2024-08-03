import sys

from subprocess import call
from data.datasets import DATASETS

import numpy as np

#del DATASETS["superconductivity"]
#del DATASETS["yacht"]
#del DATASETS["wine"]

i = int(sys.argv[1])
splits = i * 5 + np.arange(5)
num_splits = 20

for dataset, dataset_class in DATASETS.items():
    if dataset != "cpu":
        continue
    for split_id in splits:
        print(200*"=")
        print(dataset_class.name)
        print(200*"=")
        call(
            [
                "python",
                "experiments/kernel-estimation.py",
                "--experiment-name",
                f"runs-rff-cpu",
                "--dataset",
                dataset,
                "--seed-dataset",
                "1",
                "--num-splits",
                str(num_splits),
                "--split-id",
                str(split_id),
                "--max-datapoints",
                "256",
                "--lengthscale",
                "3.0",
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
                "10000",
                "--sampler-learning-rate",
                "0.001",
                "--sampler-num-steps",
                "2000",
                "--sampler-num-samples",
                "10",
                "--frame",
                "ortho",
                "--rf",
                "fourier",
            ]
        )


for dataset, dataset_class in DATASETS.items():
    if dataset != "cpu":
        continue
    for split_id in splits:
        print(200*"=")
        print(dataset_class.name)
        print(200*"=")
        call(
            [
                "python",
                "experiments/kernel-estimation.py",
                "--experiment-name",
                f"runs-rlf-cpu",
                "--dataset",
                dataset,
                "--seed-dataset",
                "1",
                "--num-splits",
                str(num_splits),
                "--split-id",
                str(split_id),
                "--max-datapoints",
                "256",
                #"--lengthscale",
                #"3.0",
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
                "10000",
                "--sampler-learning-rate",
                "0.001",
                "--sampler-num-steps",
                "4000",
                "--sampler-num-samples",
                "40",
                "--frame",
                "ortho_anti",
                "--rf",
                "laplace",
            ]
        )

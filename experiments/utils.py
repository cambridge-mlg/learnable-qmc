from typing import Tuple
import argparse
import os
from datetime import datetime

import git
import yaml
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate


def initialize_experiment() -> Tuple[DictConfig, str]:
    """Initialize experiment by parsing the config file, checking that the
    repo is clean, creating a path for the experiment, and creating a
    writer for tensorboard.

    Returns:
        experiment: experiment config object.
        path: path to experiment.
        writer: tensorboard writer.
    """

    # Make argument parser with just the config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args, config_changes = parser.parse_known_args()

    # Create a repo object and check if local repo is clean
    repo = git.Repo(search_parent_directories=True)

    # Check that the repo is clean
    assert (
        args.debug or not repo.is_dirty()
    ), "Repo is dirty, please commit changes."
    assert args.debug or not has_commits_ahead(
        repo
    ), "Repo has commits ahead, please push changes."

    # Initialize experiment, make path and writer
    OmegaConf.register_new_resolver("eval", eval)
    config = OmegaConf.load(args.config)
    config_changes = OmegaConf.from_cli(config_changes)

    config = OmegaConf.merge(config, config_changes)
    experiment_name, results_path = make_experiment_path(
        config,
        allow_overwrite=args.overwrite,
    )
    config.commit = get_current_commit_hash(repo) if not args.debug else None
    config.misc.experiment_name = experiment_name

    # Initialize experiment
    experiment = instantiate(config)

    # Write config to file together with commit hash
    with open(f"{results_path}/config.yml", "w") as file:
        config = OmegaConf.to_container(config)
        yaml.dump(config, file, indent=4, sort_keys=False)

    return experiment, config, results_path


def has_commits_ahead(repo: git.Repo) -> bool:
    """Check if there are commits ahead in the local current branch.

    Arguments:
        repo: git repo object.

    Returns:
        has_commits_ahead: True if there are commits ahead, False otherwise.
    """
    if repo.head.is_detached:
        assert not repo.is_dirty(), "Repo is dirty, please commit changes."
        return False

    else:
        current_branch = repo.active_branch.name
        commits = list(
            repo.iter_commits(f"origin/{current_branch}..{current_branch}")
        )
        return len(commits) > 0


def get_current_commit_hash(repo: git.Repo) -> str:
    """Get the current commit hash of the local repo.

    Arguments:
        repo: git repo object.

    Returns:
        commit_hash: current commit hash.
    """
    if repo.head.is_detached:
        return repo.commit(repo.head.object).hexsha

    else:
        return repo.head.commit.hexsha


def make_experiment_path(
    experiment: DictConfig, allow_overwrite: bool = False
) -> Tuple[str, str]:
    """Parse initialized experiment config and make up a path
    for the experiment, and create it if it doesn't exist,
    otherwise raise an error. Finally return the path.

    Arguments:
        config: config object.

    Returns:
        experiment_path: path to the experiment.
    """

    if not allow_overwrite:
        assert os.path.exists(experiment.misc.results_path)

    try:
        experiment_name = experiment.misc.experiment_name
    except AttributeError:
        experiment_name = datetime.now().strftime(
            f"{os.getlogin()}-%m-%d-%H-%M-%S"
        )

    results_path = os.path.join(
        experiment.misc.results_path,
        experiment_name,
    )

    if not os.path.exists(results_path):
        print(f"Making path for experiment results: {results_path}.")
        os.makedirs(results_path)

    else:
        print(f"Overwriting path for experiment results: {results_path}.")

    return experiment_name, results_path

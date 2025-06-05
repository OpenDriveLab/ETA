from carformer.data import (
    B2DSequenceDataset,
    DatasetPreloader,
    InMemoryDatasetPreloader,
)
import os


def get_datasets(config, model=None, return_all=False, splits=["train", "val"]):
    if return_all:
        return _get_entire_dataset(
            config.data_dir,
            config.training,
            config.dataset.data_format,
            preload=config.preload,
            preload_in_memory=config.preload_in_memory,
            wipe_cache=config.wipe_cache,
            cache_dir=config.cache_dir,
            model=model,
        )

    return _get_datasets(
        config.data_dir,
        config.training,
        config.dataset.data_format,
        preload=config.preload,
        preload_in_memory=config.preload_in_memory,
        wipe_cache=config.wipe_cache,
        cache_dir=config.cache_dir,
        model=model,
        splits=splits,
    )


def _get_datasets(
    data_dir,
    train_cfg,
    data_format,
    preload=False,
    preload_in_memory=False,
    wipe_cache=False,
    cache_dir="",
    model=None,
    splits=["train", "val"],
):
    # If is_plant is a boolean, give a deprecated warning
    if isinstance(data_format, bool):
        print(
            "Warning: is_plant boolean is deprecated, please use the data_format field in the dataset configuration"
        )

    if data_format == "plant":
        data_module = PlantSequenceDataset
    elif data_format == "b2d":
        data_module = B2DSequenceDataset
    elif data_format == "pdm":
        data_module = PDMSequenceDataset
    elif data_format == "tf":
        raise ValueError("Transfuser dataset not supported")
        # data_module = SequenceDataset
    else:
        raise ValueError(f"Invalid data format {data_format}")

    if "train" in splits:
        train_dataset = data_module(
            data_dir,
            train_cfg.splits.train,
            train_cfg,
        )
    else:
        train_dataset = None

    if "val" in splits:
        val_dataset = data_module(
            data_dir,
            train_cfg.splits.val,
            train_cfg,
        )
    else:
        val_dataset = None

    if preload:
        assert cache_dir != "", "Cache dir must be specified if preloading is enabled"

        preloader = (
            DatasetPreloader if not preload_in_memory else InMemoryDatasetPreloader
        )
        args = []

        if train_dataset is not None:
            train_dataset = preloader(
                train_dataset,
                os.path.join(cache_dir, train_dataset.get_parametrized_dirname()),
                *args,
                wipe_cache=wipe_cache,
            )

        if val_dataset is not None:
            val_dataset = preloader(
                val_dataset,
                os.path.join(cache_dir, val_dataset.get_parametrized_dirname()),
                *args,
                wipe_cache=wipe_cache,
            )
        if train_dataset is not None:
            train_dataset.load_state()

        if val_dataset is not None:
            val_dataset.load_state()

    return train_dataset, val_dataset


def _get_entire_dataset(
    data_dir,
    train_cfg,
    is_plant,
    preload=False,
    preload_in_memory=False,
    wipe_cache=False,
    cache_dir="",
    model=None,
):
    if data_format == "b2d":
        data_module = B2DSequenceDataset
    else:
        raise ValueError(f"Invalid data format {data_format}")

    all_dataset = data_module(
        data_dir,
        "all",
        train_cfg,
    )

    if preload:
        assert cache_dir != "", "Cache dir must be specified if preloading is enabled"

        preloader = (
            DatasetPreloader if not preload_in_memory else InMemoryDatasetPreloader
        )

        args = []

        all_dataset = preloader(
            all_dataset,
            os.path.join(cache_dir, all_dataset.get_parametrized_dirname()),
            *args,
            wipe_cache=wipe_cache,
        )

        all_dataset.load_state()

    return all_dataset

from typing import Any, Dict, List, Optional

import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.logger import Logger
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.processing_continue import (
    convert_modality_of_checkpoint_state_dct,
)
from sevenn.train.trainer import Trainer


def loader_from_config(
    config: Dict[str, Any], dataset: Dataset, is_train: bool = False
) -> DataLoader:
    batch_size = config[KEY.BATCH_SIZE]
    shuffle = is_train and config[KEY.TRAIN_SHUFFLE]
    sampler = None
    loader_args = {'dataset': dataset, 'batch_size': batch_size, 'shuffle': shuffle}
    if KEY.NUM_WORKERS in config and config[KEY.NUM_WORKERS] > 0:
        loader_args.update({'num_workers': config[KEY.NUM_WORKERS]})

    if config[KEY.IS_DDP]:
        dist.barrier()
        sampler = DistributedSampler(
            dataset, dist.get_world_size(), dist.get_rank(), shuffle=shuffle
        )
        loader_args.update({'sampler': sampler})
        loader_args.pop('shuffle')  # sampler is mutually exclusive with shuffle
    return DataLoader(**loader_args)


def train_v2(config: Dict[str, Any], working_dir: str) -> None:
    """
    Main program flow, since v0.9.6
    """
    import sevenn.train.atoms_dataset as atoms_dataset
    import sevenn.train.graph_dataset as graph_dataset
    import sevenn.train.modal_dataset as modal_dataset

    from .processing_continue import processing_continue_v2
    from .processing_epoch import processing_epoch_v2

    log = Logger()
    log.timer_start('total')

    if KEY.LOAD_TRAINSET not in config and KEY.LOAD_DATASET in config:
        log.writeline('***************************************************')
        log.writeline('For train_v2, please use load_trainset_path instead')
        log.writeline('I will assign load_trainset as load_dataset')
        log.writeline('***************************************************')
        config[KEY.LOAD_TRAINSET] = config.pop(KEY.LOAD_DATASET)

    # config updated
    start_epoch = 1
    state_dicts: Optional[List[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, start_epoch = processing_continue_v2(config)

    if config.get(KEY.USE_MODALITY, False):
        datasets = modal_dataset.from_config(config, working_dir)
    elif config[KEY.DATASET_TYPE] == 'graph':
        datasets = graph_dataset.from_config(config, working_dir)
    elif config[KEY.DATASET_TYPE] == 'atoms':
        datasets = atoms_dataset.from_config(config, working_dir)
    else:
        raise ValueError(f'Unknown dataset type: {config[KEY.DATASET_TYPE]}')
    loaders = {
        k: loader_from_config(config, v, is_train=(k == 'trainset'))
        for k, v in datasets.items()
    }

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    log.print_model_info(model, config)

    trainer = Trainer.from_config(model, config)
    if state_dicts:
        trainer.load_state_dicts(*state_dicts, strict=False)

    processing_epoch_v2(
        config, trainer, loaders, start_epoch, working_dir=working_dir
    )
    log.timer_end('total', message='Total wall time')


def train(config, working_dir: str):
    """
    Main program flow, until v0.9.5
    """
    import os

    import torch

    from .processing_continue import processing_continue
    from .processing_dataset import processing_dataset
    from .processing_epoch import processing_epoch

    log = Logger()
    log.timer_start('total')

    # config updated
    state_dicts: Optional[List[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, start_epoch, init_csv = processing_continue(config)
    else:
        start_epoch, init_csv = 1, True

    # config updated
    train, valid, _ = processing_dataset(config, working_dir)
    datasets = {'dataset': train, 'validset': valid}
    loaders = {
        k: loader_from_config(config, v, is_train=(k == 'dataset'))
        for k, v in datasets.items()
    }
    loaders = list(loaders.values())

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)

    log.write('Model building was successful\n')

    trainer = Trainer.from_config(model, config)
    if state_dicts:
        state_dicts = convert_modality_of_checkpoint_state_dct(config, state_dicts)
        trainer.load_state_dicts(*state_dicts, strict=False)

    log.print_model_info(model, config)

    Logger().write('Trainer initialized, ready to training\n')
    Logger().bar()
    log.write('Trainer initialized, ready to training\n')
    log.bar()

    # Fisher computation logic removed from legacy train()
    # Use train_fisher() instead

    processing_epoch(trainer, config, loaders, start_epoch, init_csv, working_dir)
    log.timer_end('total', message='Total wall time')


def train_fisher(config: Dict[str, Any], working_dir: str) -> None:
    """
    Fisher computation flow using v2 architecture logic.
    """
    import sevenn.train.atoms_dataset as atoms_dataset
    import sevenn.train.graph_dataset as graph_dataset
    import sevenn.train.modal_dataset as modal_dataset

    from .processing_continue import processing_continue_v2

    log = Logger()
    log.timer_start('total')

    if KEY.LOAD_TRAINSET not in config and KEY.LOAD_DATASET in config:
        config[KEY.LOAD_TRAINSET] = config.pop(KEY.LOAD_DATASET)

    # config updated
    state_dicts: Optional[List[dict]] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, _ = processing_continue_v2(config)

    if config.get(KEY.USE_MODALITY, False):
        datasets = modal_dataset.from_config(config, working_dir)
    elif config[KEY.DATASET_TYPE] == 'graph':
        datasets = graph_dataset.from_config(config, working_dir)
    elif config[KEY.DATASET_TYPE] == 'atoms':
        datasets = atoms_dataset.from_config(config, working_dir)
    else:
        raise ValueError(f'Unknown dataset type: {config[KEY.DATASET_TYPE]}')
    
    # We only need trainset for Fisher computation
    loaders = {
        k: loader_from_config(config, v, is_train=(k == 'trainset'))
        for k, v in datasets.items()
        if k == 'trainset'
    }

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    log.print_model_info(model, config)

    trainer = Trainer.from_config(model, config)
    if state_dicts:
        trainer.load_state_dicts(*state_dicts, strict=False)

    compute_fisher_information(config, trainer, loaders['trainset'], working_dir)
    log.timer_end('total', message='Total wall time')


def compute_fisher_information(config, trainer, loader, working_dir: str):
    """
    Compute Fisher information matrix and optimal parameters for EWC.
    
    Args:
        config: Training configuration
        trainer: Trainer instance
        loader: Training data loader (batch_size should be 1)
        working_dir: Working directory for saving outputs
    """
    import os

    import torch

    log = Logger()

    if config[KEY.BATCH_SIZE] != 1:
        raise ValueError('Batch size must be 1 to compute Fisher Information.')
    
    fname_fisher = os.path.join(working_dir, 'fisher_sevenn.pt')
    fname_opt = os.path.join(working_dir, 'opt_params_sevenn.pt')
    
    if os.path.isfile(fname_fisher) or os.path.isfile(fname_opt):
        raise ValueError(
            f'{fname_fisher} or {fname_opt} already exist!'
            ' Abort computation to avoid overwrite'
        )
    
    log.write('Calculating Fisher information and'
              ' optimized parameters for EWC...\n')

    loss_thr = config.get(KEY.CONTINUE, {}).get(KEY.LOSS_THR, -1)
    fisher_info, optim_param, calc_num = trainer.compute_fisher_matrix(
        loader, loss_thr
    )

    torch.save(fisher_info, fname_fisher)
    torch.save(optim_param, fname_opt)

    log.write(f'Calculation finished. {calc_num} configurations'
              ' from trainingset were used.\n')
    log.write(f'Files {fname_fisher} and {fname_opt}'
              ' are generated.\n')
    log.bar()


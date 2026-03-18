import random
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import sevenn._keys as KEY
from sevenn.logger import Logger
from sevenn.model_build import build_E3_equivariant_model
from sevenn.scripts.processing_continue import processing_continue_v2
from sevenn.train.trainer import RehearsalTrainer

from .process_dataset_rehearsal import process_dataset_rehearsal
from .processing_epoch_with_rehearsal import processing_epoch_with_rehearsal


def init_mem_loaders(memory_list, config):
    """
    Create DataLoader for memory dataset.
    
    Args:
        memory_list: List of memory data samples
        config: Training configuration
        
    Returns:
        DataLoader: Memory data loader
    """
    mem_batch_size = config[KEY.MEM_BATCH_SIZE]
    is_ddp = config.get(KEY.IS_DDP, False)

    if is_ddp:
        dist.barrier()
        mem_sampler = DistributedSampler(
            memory_list,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        memory_loader = DataLoader(
            memory_list, batch_size=mem_batch_size, sampler=mem_sampler
        )
    else:
        memory_loader = DataLoader(
            memory_list, batch_size=mem_batch_size, shuffle=True
        )
    return memory_loader


def train_rehearsal(config: Dict[str, Any], working_dir: str) -> None:
    """
    Main rehearsal training flow.
    Combines new task training with memory replay for continual learning.
    
    Args:
        config: Training configuration dictionary
        working_dir: Working directory for outputs
    """
    import sevenn.train.graph_dataset as graph_dataset
    from sevenn.scripts.train import loader_from_config

    log = Logger()
    log.timer_start('total')
    
    seed = config[KEY.RANDOM_SEED]
    random.seed(seed)
    torch.manual_seed(seed)

    # Handle checkpoint continuation
    start_epoch = 1
    state_dicts: Optional[list] = None
    if config[KEY.CONTINUE][KEY.CHECKPOINT]:
        state_dicts, start_epoch = processing_continue_v2(config)

    # Load main datasets
    if KEY.LOAD_TRAINSET not in config and KEY.LOAD_DATASET in config:
        config[KEY.LOAD_TRAINSET] = config.pop(KEY.LOAD_DATASET)
    
    datasets = graph_dataset.from_config(config, working_dir)
    loaders = {
        k: loader_from_config(config, v, is_train=(k == 'trainset'))
        for k, v in datasets.items()
    }

    # Initialize memory dataset for rehearsal
    assert config[KEY.LOAD_MEMORY_PATH], 'No memory dataset given for rehearsal'

    # process_dataset_rehearsal now returns list of graphs directly
    mem_dat_list = process_dataset_rehearsal(config, log)
    log.write(f'Memory ratio: {config[KEY.MEM_RATIO]}\n')
    log.write(f'Memory batch size: {config[KEY.MEM_BATCH_SIZE]}\n')
    
    random.shuffle(mem_dat_list)
    
    # Apply memory ratio
    mem_ratio = config.get(KEY.MEM_RATIO, 1.0)
    mem_dat_list = mem_dat_list[:int(len(mem_dat_list) * mem_ratio)]
    
    mem_loader = init_mem_loaders(mem_dat_list, config)

    log.write('\nModel building...\n')
    model = build_E3_equivariant_model(config)
    log.write('Model building was successful\n')

    # Use RehearsalTrainer instead of regular Trainer
    trainer = RehearsalTrainer.from_config(model, config)

    if state_dicts is not None:
        trainer.load_state_dicts(*state_dicts, strict=False)

    log.print_model_info(model, config)

    log.write('Trainer initialized, ready to training\n')
    log.bar()

    processing_epoch_with_rehearsal(
        trainer, config, loaders, mem_loader, start_epoch, working_dir,
    )
    log.timer_end('total', message='Total wall time')

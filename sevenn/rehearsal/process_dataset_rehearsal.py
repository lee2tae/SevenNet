import sevenn._keys as KEY
from sevenn.train.graph_dataset import SevenNetGraphDataset


def process_dataset_rehearsal(config, log):
    """
    Load memory dataset for rehearsal from load_memory_path.
    Uses the same loading mechanism as load_dataset_path for consistency.
    
    Args:
        config: Training configuration dictionary
        log: Logger instance
        
    Returns:
        List[AtomGraphData]: List of graph data for rehearsal
    """
    cutoff = config[KEY.CUTOFF]
    num_cores = config.get(KEY.PREPROCESS_NUM_CORES, 1)

    log.write('\nTry to use rehearsal from load_memory_path\n')
    
    load_memory_path = config[KEY.LOAD_MEMORY_PATH]
    if isinstance(load_memory_path, str):
        load_memory_path = [load_memory_path]
    
    graph_list = []
    for file in load_memory_path:
        log.write(f'Loading memory file: {file}\n')
        # Use the same loading method as load_dataset_path
        graphs = SevenNetGraphDataset.file_to_graph_list(
            file=file,
            cutoff=cutoff,
            num_cores=num_cores,
        )
        graph_list.extend(graphs)
    
    log.format_k_v(
        '\nLoaded memory set size is', len(graph_list), write=True
    )

    log.write('Memory set loading was successful\n')

    return graph_list

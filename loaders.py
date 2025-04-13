import torch
from torch.utils.data import DataLoader
import numpy as np
import random


def get_loaders(dataset_train, dataset_valid, dataset_test, SEED, batch_size, n_workers=5):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    gtr = torch.Generator()
    gtr.manual_seed(SEED)
    gv = torch.Generator()
    gv.manual_seed(SEED)
    gte = torch.Generator()
    gte.manual_seed(SEED)

    if n_workers > 1:
        training_loader = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            worker_init_fn=seed_worker,
            generator=gtr,
        )
        validation_loader = DataLoader(
            dataset=dataset_valid,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            worker_init_fn=seed_worker,
            generator=gv,
        )
        test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            num_workers=n_workers,
            worker_init_fn=seed_worker,
            shuffle=False,
            generator=gte,
        )
    else:
        training_loader = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            dataset=dataset_valid,
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            dataset=dataset_test,
            batch_size=1,
            shuffle=False,
        )
    return training_loader, validation_loader, test_loader

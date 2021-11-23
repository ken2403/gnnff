from torch.utils.data import DataLoader, RandomSampler

from gnnff.data.split import train_test_split


__all__ = ["get_loader"]


def get_loader(dataset, args):
    """
    Returns
    -------
    train_data, val_loader, test_loader : torch.utils.data.DataLoader
    """
    train_data, val_data, test_data = train_test_split(
        dataset, *args.split, split_path=args.modelpath
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=RandomSampler(train_data),
        num_workers=4,
        pin_memory=args.cuda,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=args.cuda,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=args.cuda,
    )
    return train_data, val_loader, test_loader

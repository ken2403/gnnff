import os
import torch
import logging
import schnetpack as spk

from gnnff.data.celldata import CellData
from gnnff.model.gnnff import GNNFF
from gnnff.utils.data import get_loader
from gnnff.utils.training import get_trainer
from gnnff.utils.script_utils import build_parser, read_from_json


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def main(args):
    # set up learning environment
    device = torch.device("cuda" if args.cuda else "cpu")

    # get dataset
    logging.info("loading the dataset...")
    dataset = CellData(db_path=args.datapath, cutoff=args.cutoff)

    # get dataloaders
    logging.info("creating the data splits...")
    train_loader, val_loader, test_loader = get_loader(dataset, args)

    # setup property metrics
    metrics = [
        spk.train.metrics.MeanAbsoluteError(args.property, args.property),
        spk.train.metrics.RootMeanSquaredError(args.property, args.property),
    ]

    # train or eval
    if args.mode == "train":
        # build model
        logging.info("building the model...")
        model = GNNFF(
            n_node_feature=args.n_node_feature,
            n_edge_feature=args.n_edge_feature,
            n_message_passing=args.n_message_passing,
            cutoff=args.cutoff,
            gaussian_filter_end=args.gaussian_filter_end,
            share_weights=args.share_weights,
            return_intermediate=args.return_intermediate,
            property=args.property,
            n_output_layers=args.n_output_layers,
        )

        # training
        logging.info("setting training configurations...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)
        logging.info("training...")
        trainer.train(
            device, n_epochs=args.n_epochs, lambda_=args.regularization_lambda
        )
        logging.info("...training done!")


if __name__ == "__main":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "from_json":
        args = read_from_json(args.jsonpath)

    main(args)

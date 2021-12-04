#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import logging
import argparse

from gnnff.data.celldata import CellData
from gnnff.model.gnnff import GNNFF
from gnnff.utils.evaluation import evaluate
from gnnff.utils.data import get_loader
from gnnff.utils.training import get_metrics, get_trainer
from gnnff.utils.script_utils import ScriptError, count_params, read_from_json


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
    metrics = get_metrics(args)

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
            update_method=args.update_method,
            share_weights=args.share_weights,
            return_intermid=False,
            properties=args.predict_property,
            n_output_layers=args.n_output_layers,
        )
        if args.parallel:
            model = nn.DataParallel(model)
        logging.info(
            "The model you built has: {} parameters".format(count_params(model))
        )

        # training
        logging.info("setting up training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)
        logging.info("training...")
        torch.backends.cudnn.benchmark = True
        trainer.train(
            device, n_epochs=args.n_epochs, lambda_=args.regularization_lambda
        )
        logging.info("...training done!")

    # train or eval
    elif args.mode == "eval":

        # remove old evaluation files
        evaluation_fp = os.path.join(args.modelpath, "evaluation.txt")
        if os.path.exists(evaluation_fp):
            os.remove(evaluation_fp)

        # load model
        logging.info("loading trained model...")
        model = torch.load(
            os.path.join(args.modelpath, "best_model"), map_location=device
        )

        # run evaluation
        logging.info("evaluating...")
        with torch.no_grad():
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
            )
        logging.info("... evaluation done!")

    else:
        raise ScriptError("Unknown mode: {}".format(args.mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("argsjsonpath", help="json file path")
    args = parser.parse_args()
    args = read_from_json(args.argsjsonpath)

    main(args)

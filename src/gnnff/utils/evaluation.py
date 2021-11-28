import os
import csv


__all__ = ["evaluate"]


def evaluate(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    metrics,
    custom_header=None,
):

    header = []
    results = []

    loaders = dict(train=train_loader, validation=val_loader, test=test_loader)
    for datasplit in args.split:
        header += ["{} MAE".format(datasplit), "{} RMSE".format(datasplit)]
        if args.parallel:
            model.to(f"cuda:{model.device_ids[0]}")

            state_dict = model.state_dict()
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                v.to(f"cuda:{model.device_ids[0]}")
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            device = f"cuda:{model.device_ids[0]}"
            derivative = model.module.output_modules[0].derivative
        if not args.parallel:
            derivative = model.output_modules[0].derivative
        if derivative is not None:
            header += [
                "{} MAE ({})".format(datasplit, derivative),
                "{} RMSE ({})".format(datasplit, derivative),
            ]
        results += evaluate_dataset(metrics, model, loaders[datasplit], device)

    if custom_header:
        header = custom_header

    eval_file = os.path.join(args.modelpath, "evaluation.txt")
    with open(eval_file, "w") as file:
        wr = csv.writer(file)
        wr.writerow(header)
        wr.writerow(results)


def evaluate_dataset(metrics, model, loader, device):
    model.eval()

    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [metric.aggregate() for metric in metrics]
    return results

from schnetpack.train.hooks import Hook


__all__ = ["NaNStoppingHook"]


class NaNStopError(Exception):
    pass


class NaNStoppingHook(Hook):
    def on_batch_end(self, trainer, train_batch, result, loss):
        if loss.isnan().any():
            trainer._stop = True
            raise NaNStopError(
                "The value of training loss has become nan! Stop training."
            )

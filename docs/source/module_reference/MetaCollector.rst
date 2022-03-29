MetaCollector
==============================================

MetaCollector is the key component for collect data for evaluation in meta learning circumstance.

Overall, we extend ``Collector`` to ``MetaCollector``.
The extended modification can be listed briefly as following:

- **[Override]** ``self.eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor)``: Collect data for evaluation.

- **[Override]** ``self.data_collect(self, train_data)``: Collect the evaluation resource from training data.
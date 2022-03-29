MetaTrainer
==============================================

MetaTrainer is the key component for training a meta learning method.

Overall, we extend :attr:`Trainer` to :attr:`MetaTrainer`.
If you want to implement a meta learning model, please extend this class and implement ``_train_epoch()`` method. eg. You can create :attr:`MeLUTrainer(MetaTrainer)` and implement its ``_train_epoch()`` method.

The extended modification can be listed briefly as following:

- **[Override]** ``self.evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False)``: Evaluation.

- **[Abstract]** ``self.taskDesolve(task)``: Desolve a task into spt_x,spt_y,qrt_x,qrt_y.

- **[Abstract]** ``self._train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False)``: An epoch of training.

self.evaluate(eval_data, load_best_model=True, model_file=None, show_progress=False)
-------------------------

We adapt the evaluation process with task circumstance in meta learning.

self.taskDesolve(task)
-------------------------

This is an abstract method which is waiting for specific model to implement.
It desolves a task into spt_x,spt_y,qrt_x,qrt_y.

:task(Task): The object of class :attr:`Task`
:[return] spt_x,spt_y,qrt_x,qrt_y: The four base parts of task in meta learning.

self._train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)
-------------------------

This is an abstract method which is waiting for specific model to implement.
This method indicates for an epoch of training.
It can be called by ``self.fit()``.
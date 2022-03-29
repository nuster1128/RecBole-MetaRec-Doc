MetaRecommender
==============================================

MetaRecommender is the key component for implementing meta learning model.
It is an abstract recommender for meta learning, in order to clearify ``Task`` for meta learning.

Overall, we extend :attr:`AbstractRecommender` to :attr:`MetaRecommender`.
If you want to implement a meta learning model, please extend this class and implement ``calculate_loss()`` method and ``predict()`` method. eg. You can create :attr:`MeLU(MetaRecommender)` and implement its ``calculate_loss()`` and ``predict()`` method.

The extended modification can be listed briefly as following:

- **[Abstract]** ``self.calculate_loss(taskBatch)``: Calculate the loss or the grad of the batch of tasks.

- **[Abstract]** ``self.predict(spt_x,spt_y,qrt_x)``: Predict the score of the query set of the task.


self.calculate_loss(taskBatch)
-------------------------

This is an abstract method which is waiting for specific model to implement.

Calculate the loss or the grad of the batch of tasks.
Some meta learning model uses loss to backward.
And some meta learning model uses grad for further calculation.

:tasks(List of Task): The list of tasks.
:[return] loss(torch.Tensor),grad(torch.Tensor): Training loss and grad of the batch of tasks. One of them can be None.

self.predict(spt_x,spt_y,qrt_x)
-------------------------

This is an abstract method which is waiting for specific model to implement.

Predict the score of the query set of the task.

:spt_x(Interaction): Input of spt.
:spt_y(torch.Tensor): Rating/Label of spt. shape: ``[batchsize, 1]``
:qrt_x(Interaction): Input of qrt.
:[return] scores(torch.Tensor): The predicted scores of the query set of the task. shape: ``[batchsize, 1]``
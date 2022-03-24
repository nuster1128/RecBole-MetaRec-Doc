recbole.MetaModule.MetaDataLoader
==============================================

MetaDataLoader is the key component for transforming dataset into task form.
As usual, we consider each user as a task.
Here, a batch of data refers to a batch of tasks.

Overall, we extend :attr:`AbstractDataLoader` to :attr:`MetaDataLoader`.
The extended modification can be listed briefly as following:

- **[Add]** ``self.transformToTaskFormat()``: Generate `Task` dict from dataset.

- **[Add]** ``self.generateSingleTaskForTrain(uid,v)``: Format a single task.

- **[Add]** ``self.getUserList()``: Generate user list of this dataset.

- **[Override]** ``self._init_batch_size_and_step()``: Initialize ``train_batch_size``.

- **[Override]** ``self.pr_end()``: Get the number of tasks.

- **[Override]** ``self._shuffle()``: Shuffle the task.

- **[Override]** ``self._next_batch_data()``: Generate a batch of tasks iteratively.


self.transformToTaskFormat()
-------------------------

This function is used to generate ``task`` dict from dataset.
It will return 'taskDict' for this MetaDataLoader.
During the process of this method, it will call ``self.generateSingleTaskForTrain(uid,v)`` to deal with a single ``task(user)``.

:[return] finalTaskDict(dict): A dict whose keys are ``user_id`` and values are corresponding :attr:`Task` object.

self.generateSingleTaskForTrain(uid,v)
-------------------------

We use this function to generate a task.

:uid: uid from function ``transformToTaskFormat()``.
:v: value from function ``transformToTaskFormat()``.
:[return]: An object of class Task.

self.getUserList()
-------------------------

This function can generate user list of this dataset.

:[return] userlist (1D numpy.ndarray): an 1D array of user list.

self._init_batch_size_and_step()
-------------------------

This function is used to initialize ``train_batch_size``.
The ``train_batch_size`` indicates the number of tasks for each training batch.

self.pr_end()
-------------------------

Get the number of ``tasks(users)``.

self._shuffle()
-------------------------

Shuffle the task.

self._next_batch_data()
-------------------------

This function is used to generate a batch of tasks iteratively.
:[return] taskBatch(list): A list of task. The length is ``train_batch_size``.
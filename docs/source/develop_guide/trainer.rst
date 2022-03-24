Customize Trainer
==============================================
Here, we will show how to customize your model training and testing process.

Import packages
-------------------------
First, import the packages that you need.

.. code:: python

    from tqdm import tqdm
    from copy import deepcopy
    import torch
    from collections import OrderedDict
    from recbole.utils import FeatureSource, set_color
    from recbole.data.interaction import Interaction
    from recbole.utils import get_gpu_usage
    from recbole.MetaModule.MetaTrainer import MetaTrainer

Create your trainer class
-------------------------
Second, create your model by extending :attr:`MetaTrainer` and initialize it.

.. code:: python

    class MeLUTrainer(MetaTrainer):
        def __init__(self,config,model):
            super(MeLUTrainer, self).__init__(config,model)

            self.lr = config['melu_args']['lr']
            self.xFields = model.dataset.fields(source=[FeatureSource.USER, FeatureSource.ITEM])
            self.yField = model.RATING

Implement taskDesolve method
-------------------------
Third, implement ``self.taskDesolve(task)`` method.

.. code:: python

    def taskDesolve(self,task):
        spt_x,qrt_x=OrderedDict(),OrderedDict()
        for field in self.xFields:
            spt_x[field]=task.spt[field]
            qrt_x[field]=task.qrt[field]
        spt_y=task.spt[self.yField]
        qrt_y=task.qrt[self.yField]

        spt_x, qrt_x=Interaction(spt_x),Interaction(qrt_x)
        return spt_x, spt_y, qrt_x, qrt_y

Implement _train_epoch method
-------------------------
Fourth, implement ``self._train_epoch(train_data, epoch_idx, loss_func=None, show_progress=False)`` method.

.. code:: python

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        totalLoss=torch.tensor(0.0)
        for batch_idx, taskBatch in enumerate(iter_data):
            loss, grad = self.model.calculate_loss(taskBatch)
            totalLoss+=loss

            # This is SGD process.
            newParams=OrderedDict()
            for name,params in self.model.state_dict().items():
                newParams[name]=params-self.lr*grad[name]

            self.model.load_state_dict(newParams)

            self.model.keepWeightParams = deepcopy(self.model.model.state_dict())

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return totalLoss/(batch_idx+1)

[Optional] Implement other methods
-------------------------
[Optional] Finally, implement other methods that you need.

**The complete code is as following.**

.. code:: python

    from tqdm import tqdm
    from copy import deepcopy
    import torch
    from collections import OrderedDict
    from recbole.utils import FeatureSource, set_color
    from recbole.data.interaction import Interaction
    from recbole.utils import get_gpu_usage
    from MetaModule.MetaTrainer import MetaTrainer

    class MeLUTrainer(MetaTrainer):
        def __init__(self,config,model):
            super(MeLUTrainer, self).__init__(config,model)

            self.lr = config['melu_args']['lr']
            self.xFields = model.dataset.fields(source=[FeatureSource.USER, FeatureSource.ITEM])
            self.yField = model.RATING

        def taskDesolve(self,task):
            spt_x,qrt_x=OrderedDict(),OrderedDict()
            for field in self.xFields:
                spt_x[field]=task.spt[field]
                qrt_x[field]=task.qrt[field]
            spt_y=task.spt[self.yField]
            qrt_y=task.qrt[self.yField]

            spt_x, qrt_x=Interaction(spt_x),Interaction(qrt_x)
            return spt_x, spt_y, qrt_x, qrt_y

        def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
            self.model.train()
            iter_data = (
                tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                ) if show_progress else train_data
            )
            totalLoss=torch.tensor(0.0)
            for batch_idx, taskBatch in enumerate(iter_data):
                loss, grad = self.model.calculate_loss(taskBatch)
                totalLoss+=loss

                # This is SGD process.
                newParams=OrderedDict()
                for name,params in self.model.state_dict().items():
                    newParams[name]=params-self.lr*grad[name]

                self.model.load_state_dict(newParams)

                self.model.keepWeightParams = deepcopy(self.model.model.state_dict())

                if self.gpu_available and show_progress:
                    iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

            return totalLoss/(batch_idx+1)
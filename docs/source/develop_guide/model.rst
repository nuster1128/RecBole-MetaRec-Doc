Customize Model
==============================================
Here, we will show how to customize your model structure.

Import packages
-------------------------
First, import the packages that you need.

.. code:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy
    from collections import OrderedDict
    from recbole.model.layers import MLPLayers
    from recbole.utils import InputType, FeatureSource, FeatureType
    from MetaRecommender import MetaRecommender
    from MetaUtils import GradCollector,EmbeddingTable

Create your model class
-------------------------
Second, create your model by extending :attr:`MetaRecommender` and initialize it.

.. code:: python

    class MeLU(MetaRecommender):
        input_type = InputType.POINTWISE

        def __init__(self,config,dataset):
            super(MeLU, self).__init__(config,dataset)

            self.MLPHiddenSize = config['mlp_hidden_size']
            self.localLr = config['melu_args']['local_lr']

            self.model=nn.Sequential(
                MLPLayers(self.MLPHiddenSize),
                nn.Linear(self.MLPHiddenSize[-1],1)
            )

            self.embeddingTable=EmbeddingTable(self.embedding_size,self.dataset)
            self.metaGradCollector=GradCollector(list(self.state_dict().keys()))

            self.keepWeightParams = deepcopy(self.model.state_dict())

Implement calculate_loss method
-------------------------
Third, implement ``self.calculate_loss(taskBatch)`` method.

.. code:: python

    def calculate_loss(self, taskBatch):
        totalLoss=torch.tensor(0.0)
        for task in taskBatch:
            spt_x, spt_y, qrt_x, qrt_y=self.taskDesolveEmb(task)
            self.keepWeightParams = deepcopy(self.model.state_dict())   # Params into cache
            qrt_y_predict=self.forward(spt_x,spt_y,qrt_x)
            loss=F.mse_loss(qrt_y_predict,qrt_y)

            grad=torch.autograd.grad(loss,self.parameters())

            self.metaGradCollector.addGrad(grad)
            totalLoss+=loss.detach()

            self.model.load_state_dict(self.keepWeightParams)           # Params back
        self.metaGradCollector.averageGrad(self.config['train_batch_size'])
        totalLoss/=self.config['train_batch_size']
        return totalLoss,self.metaGradCollector.dumpGrad()

Implement predict method
-------------------------
Fourth, implement ``self.predict(spt_x,spt_y,qrt_x)`` method.

.. code:: python

    def predict(self, spt_x,spt_y,qrt_x):
        self.keepWeightParams = deepcopy(self.model.state_dict())

        spt_x = self.embeddingTable.embeddingAllFields(spt_x)
        spt_y = spt_y.view(-1, 1)
        qrt_x = self.embeddingTable.embeddingAllFields(qrt_x)

        predict_qrt_y=self.forward(spt_x,spt_y,qrt_x)
        self.model.load_state_dict(self.keepWeightParams)

        return predict_qrt_y

[Optional] Implement other methods
-------------------------
[Optional] Finally, implement other methods that you need.

.. code:: python

    def taskDesolveEmb(self,task):
        spt_x=self.embeddingTable.embeddingAllFields(task.spt)
        spt_y = task.spt[self.RATING].view(-1, 1)
        qrt_x = self.embeddingTable.embeddingAllFields(task.qrt)
        qrt_y = task.qrt[self.RATING].view(-1, 1)
        return spt_x,spt_y,qrt_x,qrt_y

    def fieldsEmb(self,interaction):
        return self.embeddingTable.embeddingAllFields(interaction)

    def forward(self,spt_x,spt_y,qrt_x):
        originWeightParams = list(self.model.state_dict().values())
        paramNames = self.model.state_dict().keys()
        fastWeightParams=OrderedDict()

        spt_y_predict=self.model(spt_x)
        localLoss=F.mse_loss(spt_y_predict,spt_y)
        self.model.zero_grad()
        grad=torch.autograd.grad(localLoss,self.model.parameters(),create_graph=True,retain_graph=True)

        for index,name in enumerate(paramNames):
            fastWeightParams[name]=originWeightParams[index]-self.localLr*grad[index]

        self.model.load_state_dict(fastWeightParams)        #Simplify to FOMAML @Nuster
        qrt_y_predict=self.model(qrt_x)

        return qrt_y_predict


**The complete code is as following.**

.. code:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from copy import deepcopy
    from collections import OrderedDict
    from recbole.model.layers import MLPLayers
    from recbole.utils import InputType, FeatureSource, FeatureType
    from MetaRecommender import MetaRecommender
    from MetaUtils import GradCollector,EmbeddingTable

    class MeLU(MetaRecommender):
        input_type = InputType.POINTWISE

        def __init__(self,config,dataset):
            super(MeLU, self).__init__(config,dataset)

            self.MLPHiddenSize = config['mlp_hidden_size']
            self.localLr = config['melu_args']['local_lr']

            self.model=nn.Sequential(
                MLPLayers(self.MLPHiddenSize),
                nn.Linear(self.MLPHiddenSize[-1],1)
            )

            self.embeddingTable=EmbeddingTable(self.embedding_size,self.dataset)
            self.metaGradCollector=GradCollector(list(self.state_dict().keys()))

            self.keepWeightParams = deepcopy(self.model.state_dict())

        def taskDesolveEmb(self,task):
            spt_x=self.embeddingTable.embeddingAllFields(task.spt)
            spt_y = task.spt[self.RATING].view(-1, 1)
            qrt_x = self.embeddingTable.embeddingAllFields(task.qrt)
            qrt_y = task.qrt[self.RATING].view(-1, 1)
            return spt_x,spt_y,qrt_x,qrt_y

        def fieldsEmb(self,interaction):
            return self.embeddingTable.embeddingAllFields(interaction)

        def forward(self,spt_x,spt_y,qrt_x):
            originWeightParams = list(self.model.state_dict().values())
            paramNames = self.model.state_dict().keys()
            fastWeightParams=OrderedDict()

            spt_y_predict=self.model(spt_x)
            localLoss=F.mse_loss(spt_y_predict,spt_y)
            self.model.zero_grad()
            grad=torch.autograd.grad(localLoss,self.model.parameters(),create_graph=True,retain_graph=True)

            for index,name in enumerate(paramNames):
                fastWeightParams[name]=originWeightParams[index]-self.localLr*grad[index]

            self.model.load_state_dict(fastWeightParams)        #Simplify to FOMAML @Nuster
            qrt_y_predict=self.model(qrt_x)

            return qrt_y_predict

        def calculate_loss(self, taskBatch):
            totalLoss=torch.tensor(0.0)
            for task in taskBatch:
                spt_x, spt_y, qrt_x, qrt_y=self.taskDesolveEmb(task)
                self.keepWeightParams = deepcopy(self.model.state_dict())   # Params into cache
                qrt_y_predict=self.forward(spt_x,spt_y,qrt_x)
                loss=F.mse_loss(qrt_y_predict,qrt_y)

                grad=torch.autograd.grad(loss,self.parameters())

                self.metaGradCollector.addGrad(grad)
                totalLoss+=loss.detach()

                self.model.load_state_dict(self.keepWeightParams)           # Params back
            self.metaGradCollector.averageGrad(self.config['train_batch_size'])
            totalLoss/=self.config['train_batch_size']
            return totalLoss,self.metaGradCollector.dumpGrad()

        def predict(self, spt_x,spt_y,qrt_x):
            self.keepWeightParams = deepcopy(self.model.state_dict())

            spt_x = self.embeddingTable.embeddingAllFields(spt_x)
            spt_y = spt_y.view(-1, 1)
            qrt_x = self.embeddingTable.embeddingAllFields(qrt_x)

            predict_qrt_y=self.forward(spt_x,spt_y,qrt_x)
            self.model.load_state_dict(self.keepWeightParams)

            return predict_qrt_y

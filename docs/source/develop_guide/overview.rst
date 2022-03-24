Overview
==============================================
This section will present how to develop your new model.

**What you should do are:**

- Customize Configuration: Write the extra parameters of your model in the configuration file. eg. :attr:`MeLU.yaml`

- Customize Model: Extend :attr:`MetaRecommender` class to implement your model. eg. :attr:`MeLU(MetaRecommender)`

- Customize Trainer: Extend :attr:`MetaTrainer` class to implement the training process of your model. eg. :attr:`MeLUTrainer(MetaRecommender)`

MetaUtils
==============================================

class Task()
-------------------------

Task is the basis of meta learning.
For user cold start recsys, a task usually refers to a user.

def create_meta_dataset(config)
-------------------------

This function is rewritten from ``recbole.data.create_meta_dataset(config)``

def meta_data_preparation(config, dataset)
-------------------------

This function is rewritten from ``recbole.data.data_preparation(config, dataset)``

class GradCollector()
-------------------------

This is a common data struct to collect grad.

For the sake of complex calculation graph in meta learning, we construct this data struct to do grad operations on batch data.

class EmbeddingTable(nn.Module)
-------------------------

This is a data struct to embedding interactions.
It supports ``token`` and ``float`` type.
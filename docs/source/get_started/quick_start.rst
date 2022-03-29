Quick Start
==============================================

Actually, the quick start is totally as same as that in `RecBole Quick Start <https://recbole.io/docs/get_started/quick_start.html#>`_. But we still provide you a runnable demo for this module usage.

We will show you how to train and test ``MeLU`` model on the ``ml-100k-local`` dataset.

First, you can find the demo file ``run.py`` in the repository.

.. code:: python

    from recbole.utils import init_logger, init_seed
    from recbole.config import Config
    from MetaUtils import *
    from model.MeLU.MeLUTrainer import MeLUTrainer
    from model.MeLU.MeLU import MeLU

    if __name__ == '__main__':
        config = Config(model=MeLU, dataset='ml-100k-local',config_file_list=['model/MeLU/MeLU.yaml'])
        init_seed(config['seed'], config['reproducibility'])

        # logger initialization
        init_logger(config)
        logger = getLogger()
        logger.info(config)

        # dataset filtering
        dataset = create_meta_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = meta_data_preparation(config, dataset)

        # model loading and initialization
        model = MeLU(config, train_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = MeLUTrainer(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

        # model evaluation
        test_result = trainer.evaluate(test_data)

        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))

Then run the file ``run.py`` directly after RecBole installation.
And the output will be as following.

.. code:: python

    24 Mar 15:54    INFO
    General Hyper Parameters:
    gpu_id = 0
    use_gpu = True
    seed = 2020
    ......
    24 Mar 15:54    INFO  Loading model structure and parameters from saved\MeLU-Mar-24-2022_15-54-35.pth
    24 Mar 15:54    INFO  best valid result: OrderedDict([('ndcg@5', 0.5257)])
    24 Mar 15:54    INFO  test result: OrderedDict([('ndcg@5', 0.5585)])




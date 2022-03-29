.. RecBole-MetaRec documentation master file.
.. title:: RecBole-MetaRec v1.0.1
.. image:: logo.png


==============================================

Introduction
-------------------------
RecBole-MetaRec is an extended module for RecBole, which aims to help researches to compare and develop their own models in meta learning recommendation field.

This module is totally developed based on RecBole by adding extened classes and functions, without modifying any codes of RecBole core.

The contributions are briefly listed as follows:

- We extend :attr:`MetaDataset` from :attr:`Dataset` to split dataset by 'task'.
- We extend :attr:`MetaDataLoader` from :attr:`AbstractDataLoader` to transform dataset into task form.
- We extend :attr:`MetaRecommender` from :attr:`AbstractRecommender` to provide a base recommender for implementing meta learning model.
- We extend :attr:`MetaTrainer` from  :attr:`Trainer` to provide a base trainer for implementing meta learning training process.
- We extend :attr:`MetaCollector` from :attr:`Collector` to collect data for evaluation in meta learning circumstance.
- We implement :attr:`MetaUtils` with some useful toolkits for meta learning.

Therefore, researches can:

- Conveniently develop their own meta learning recommendation models.
- Conveniently learn and compare meta learning recommendation models that we have implemented.
- Enjoy advantages and features of RecBole.

**Note:** Before starting, it is strongly recommended to realize how RecBole works, and the homepage of RecBole is [https://recbole.io].

The construction is as following.

.. image:: graph.png



.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install
   get_started/quick_start
   get_started/versions

.. toctree::
   :maxdepth: 1
   :caption: Models

   models/melu

.. toctree::
   :maxdepth: 1
   :caption: Develop Guide

   develop_guide/overview
   develop_guide/configuration
   develop_guide/model
   develop_guide/trainer

.. toctree::
   :maxdepth: 1
   :caption: Module Reference

   module_reference/MetaDataset
   module_reference/MetaDataLoader.rst
   module_reference/MetaRecommender.rst
   module_reference/MetaTrainer.rst
   module_reference/MetaCollector.rst
   module_reference/MetaUtils.rst
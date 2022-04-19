Versions
==============================================

Version 1.0.0
-------------------------

- Finish the basic components and documents of RecBole-MetaModule.

Version 1.0.1
-------------------------

- Separate the RecBole-MetaModule from RecBole and renamed as RecBole-MetaRec.
- Separate the ``MetaCollector`` from ``MetaUtils`` as an individual module of RecBole-MetaRec.
- Rename ``MetaDataLoader.getUserList()`` as ``MetaDataLoader.getTaskIdList()``.
- Optimize the package structure.
- Update the document after adjustment above.

Version 1.0.2
-------------------------

- Implement models: ``MAML``, ``FOMAML``, ``MAMO``, ``TaNP``, ``LWA``, ``NLBA``, ``MetaEmb``, ``MWUF``.
- Optimize some ``MetaDataLoader`` and some utils for GPU support.
- Support token_seq embedding with ``EmbeddingTable``.
- Update dataset formats.
- Update the document after adjustment above.
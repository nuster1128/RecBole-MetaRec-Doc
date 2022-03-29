MetaDataset
==============================================

MetaDataset is the key component for splitting dataset by ``task``.

Overall, we extend :attr:`Dataset` to :attr:`MetaDataset`.

The extended modification can be listed briefly as following:

- **[Override]** ``self.bulid()``: Add ``task`` keyword for ``group_by``.

- **[Add]** ``self.split_by_ratio_meta(ratios, group_by)``: Split method by ``task``.

self.build()
-------------------------

In metaDataset, we add a new ``eval_args.group_by`` keyword ``task`` in it, which can split the dataset by user clusters.

If we set the keywords ``task`` and ``RS``, it will call ``self.split_by_ratio_meta()``, which we design it for the split method.

self.split_by_ratio_meta(ratios, group_by)
-------------------------

This function is used to split dataset by non-intersection users clusters, which means the interactions in the output datasets ``(eg. training dataset, valid dataset and test dataset)`` have non-intersection user clusters.

Split by non-intersection users is significant for user cold start task in meta learning recommendation.

:ratios: The split ratios of user clusters.
:group_by: Commonly ``task``, and only by ``task`` can this function be called.
:[return] next_ds: ``[dataset_1, dataset_2 ... dataset_n]``
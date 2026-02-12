"""BatchHandler factory. Builds BatchHandler objects from batch queues and
samplers."""

import logging
from typing import TYPE_CHECKING, Optional, Union

from sup3r.preprocessing.batch_queues.base import SingleBatchQueue
from sup3r.preprocessing.batch_queues.conditional import (
    QueueMom1,
    QueueMom1SF,
    QueueMom2,
    QueueMom2Sep,
    QueueMom2SepSF,
    QueueMom2SF,
)
from sup3r.preprocessing.batch_queues.dual import DualBatchQueue
from sup3r.preprocessing.collections.stats import StatsCollection
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.cc import DualSamplerCC
from sup3r.preprocessing.samplers.dual import DualSampler
from sup3r.preprocessing.utilities import (
    check_signatures,
    get_class_kwargs,
    log_args,
)

if TYPE_CHECKING:
    from sup3r.preprocessing.base import Container

logger = logging.getLogger(__name__)


def BatchHandlerFactory(
    MainQueueClass, SamplerClass, ValQueueClass=None, name='BatchHandler'
):
    """BatchHandler factory. Can build handlers from different queue classes
    and sampler classes. For example, to build a standard
    :class:`.BatchHandler` use
    :class:`~sup3r.preprocessing.batch_queues.SingleBatchQueue` and
    :class:`~sup3r.preprocessing.samplers.Sampler`. To build a
    :class:`~.DualBatchHandler` use
    :class:`~sup3r.preprocessing.batch_queues.DualBatchQueue` and
    :class:`~sup3r.preprocessing.samplers.DualSampler`. To build a
    :class:`~..dc.BatchHandlerDC` use a
    :class:`~sup3r.preprocessing.batch_queues.BatchQueueDC`,
    :class:`~sup3r.preprocessing.batch_queues.ValBatchQueueDC` and
    :class:`~sup3r.preprocessing.samplers.SamplerDC`

    Note
    ----
    (1) BatchHandlers include a queue for training samples and a queue for
    validation samples.
    (2) There is no need to generate "Spatial" batch handlers. Using
    :class:`Sampler` objects with a single time step in the sample shape will
    produce batches without a time dimension.
    """

    class BatchHandler(MainQueueClass):
        """BatchHandler object built from two lists of
        :class:`~sup3r.preprocessing.base.Container` objects, one with
        training data and one with validation data. These lists will be used
        to initialize lists of class:`Sampler` objects that will then be used
        to build batches at run time.

        Notes
        -----
        These lists of containers can contain data from the same underlying
        data source (e.g. CONUS WTK) (e.g. initialize train / val containers
        with different time period and / or regions, or they can be used to
        sample from completely different data sources (e.g. train on CONUS WTK
        while validating on Canada WTK).

        See Also
        --------
        :class:`~sup3r.preprocessing.samplers.Sampler`,
        :class:`~sup3r.preprocessing.batch_queues.abstract.AbstractBatchQueue`,
        :class:`~sup3r.preprocessing.collections.StatsCollection`
        """

        TRAIN_QUEUE = MainQueueClass
        VAL_QUEUE = ValQueueClass or MainQueueClass
        SAMPLER = SamplerClass

        __name__ = name

        @log_args
        def __init__(
            self,
            train_containers: list['Container'],
            val_containers: Optional[list['Container']] = None,
            sample_shape: Optional[tuple] = None,
            batch_size: int = 16,
            n_batches: int = 64,
            s_enhance: int = 1,
            t_enhance: int = 1,
            means: Optional[Union[dict, str]] = None,
            stds: Optional[Union[dict, str]] = None,
            queue_cap: Optional[int] = None,
            transform_kwargs: Optional[dict] = None,
            max_workers: int = 1,
            mode: str = 'lazy',
            feature_sets: Optional[dict] = None,
            **kwargs,
        ):
            """
            Parameters
            ----------
            train_containers : list[Container]
                List of objects with a `.data` attribute, which will be used
                to initialize Sampler objects and then used to initialize a
                batch queue of training data. The data can be a Sup3rX or
                Sup3rDataset object.
            val_containers : list[Container]
                List of objects with a `.data` attribute, which will be used
                to initialize Sampler objects and then used to initialize a
                batch queue of validation data. The data can be a Sup3rX or a
                Sup3rDataset object.
            batch_size : int
                Number of samples to get to build a single batch. A sample of
                (sample_shape[0], sample_shape[1], batch_size *
                sample_shape[2]) is first selected from underlying dataset
                and then reshaped into (batch_size, *sample_shape) to get a
                single batch. This is more efficient than getting N =
                batch_size samples and then stacking.
            n_batches : int
                Number of batches in an epoch, this sets the iteration limit
                for this object.
            s_enhance : int
                Integer factor by which the spatial axes is to be enhanced.
            t_enhance : int
                Integer factor by which the temporal axes is to be enhanced.
            means : str | dict | None
                Usually a file path for loading / saving results, or None for
                just calculating stats and not saving. Can also be a dict.
            stds : str | dict | None
                Usually a file path for loading / saving results, or None for
                just calculating stats and not saving. Can also be a dict.
            queue_cap : int
                Maximum number of batches the batch queue can store. Changing
                this can effect the speed with which batches move through
                training.
            transform_kwargs : Union[dict, None]
                Dictionary of kwargs to be passed to `self.transform`. This
                method performs smoothing / coarsening.
            max_workers : int
                Number of workers / threads to use for getting batches to fill
                queue
            mode : str
                Loading mode. Default is 'lazy', which only loads data into
                memory as batches are queued. 'eager' will load all data into
                memory right away.
            feature_sets : Optional[dict]
                Optional dictionary describing how the full set of features is
                split between `lr_only_features` and `hr_exo_features`.

                features : list | tuple
                    List of full set of features to use for sampling. If no
                    entry is provided then all data_vars from container data
                    will be used.
                lr_only_features : list | tuple
                    List of feature names or patt*erns that should only be
                    included in the low-res training set and not the high-res
                    observations. This
                hr_exo_features : list | tuple
                    List of feature names or patt*erns that should be included
                    in the high-resolution observation but not expected to be
                    output from the generative model. An example is high-res
                    topography that is to be injected mid-network.
            kwargs : dict
                Additional keyword arguments for BatchQueue and / or Samplers.
                This can vary depending on the type of BatchQueue / Sampler
                given to the Factory. For example, to build a
                :class:`~sup3r.preprocessing.batch_handlers.BatchHandlerDC`
                object (data-centric batch handler) we use a queue and sampler
                which takes spatial and temporal weight / bin arguments used to
                determine how to weigh spatiotemporal regions when sampling.
                Using
                :class:`~sup3r.preprocessing.batch_queues.ConditionalBatchQueue`
                will result in arguments for computing moments from batches and
                how to pad batch data to enable these calculations.
            """  # pylint: disable=line-too-long

            check_signatures((self.TRAIN_QUEUE, self.VAL_QUEUE, self.SAMPLER))

            add_kwargs = {
                's_enhance': s_enhance,
                't_enhance': t_enhance,
                **kwargs,
            }

            sampler_kwargs = get_class_kwargs(SamplerClass, add_kwargs)
            val_kwargs = get_class_kwargs(self.VAL_QUEUE, add_kwargs)
            main_kwargs = get_class_kwargs(MainQueueClass, add_kwargs)

            bad_kwargs = (
                set(kwargs)
                - set(sampler_kwargs)
                - set(val_kwargs)
                - set(main_kwargs)
            )

            msg = (
                f'{self.__class__.__name__} received bad '
                f'kwargs = {bad_kwargs}.'
            )
            assert not bad_kwargs, msg

            train_samplers, val_samplers = self.init_samplers(
                train_containers,
                val_containers,
                sample_shape=sample_shape,
                feature_sets=feature_sets,
                batch_size=batch_size,
                mode=mode,
                sampler_kwargs=sampler_kwargs,
            )

            logger.info('Normalizing training samplers')
            stats = StatsCollection(
                containers=train_samplers,
                means=means,
                stds=stds,
            )
            self.means = stats.means
            self.stds = stats.stds

            if not val_samplers:
                self.val_data: Union[list, type[self.VAL_QUEUE]] = []
            else:
                logger.info('Normalizing validation samplers.')
                stats.normalize(val_samplers)
                self.val_data = self.VAL_QUEUE(
                    samplers=val_samplers,
                    n_batches=n_batches,
                    thread_name='validation',
                    batch_size=batch_size,
                    queue_cap=queue_cap,
                    transform_kwargs=transform_kwargs,
                    max_workers=max_workers,
                    mode=mode,
                    **val_kwargs,
                )
            super().__init__(
                samplers=train_samplers,
                n_batches=n_batches,
                batch_size=batch_size,
                queue_cap=queue_cap,
                transform_kwargs=transform_kwargs,
                max_workers=max_workers,
                mode=mode,
                **main_kwargs,
            )

        _skip_params = ('samplers', 'data', 'containers', 'thread_name')
        _signature_objs = (__init__, SAMPLER, VAL_QUEUE, TRAIN_QUEUE)

        def init_samplers(
            self,
            train_containers,
            val_containers,
            sample_shape,
            feature_sets,
            batch_size,
            mode,
            sampler_kwargs,
        ):
            """Initialize samplers from given data containers."""
            train_samplers = [
                self.SAMPLER(
                    data=container,
                    sample_shape=sample_shape,
                    feature_sets=feature_sets,
                    batch_size=batch_size,
                    mode=mode,
                    **sampler_kwargs,
                )
                for container in train_containers
            ]
            val_samplers = (
                []
                if val_containers is None
                else [
                    self.SAMPLER(
                        data=container,
                        sample_shape=sample_shape,
                        feature_sets=feature_sets,
                        batch_size=batch_size,
                        mode=mode,
                        **sampler_kwargs,
                    )
                    for container in val_containers
                ]
            )
            return train_samplers, val_samplers

        def start(self):
            """Start the val data batch queue in addition to the train batch
            queue."""
            if hasattr(self.val_data, 'start'):
                self.val_data.start()
            super().start()

        def stop(self):
            """Stop the val data batch queue in addition to the train batch
            queue."""
            self._training_flag.clear()
            if self.val_data != []:
                self.val_data._training_flag.clear()
            if hasattr(self.val_data, 'stop'):
                self.val_data.stop()
            super().stop()

    return BatchHandler


BatchHandler = BatchHandlerFactory(
    SingleBatchQueue, Sampler, name='BatchHandler'
)
DualBatchHandler = BatchHandlerFactory(
    DualBatchQueue, DualSampler, name='DualBatchHandler'
)

BatchHandlerCC = BatchHandlerFactory(
    DualBatchQueue, DualSamplerCC, name='BatchHandlerCC'
)
BatchHandlerMom1 = BatchHandlerFactory(
    QueueMom1, Sampler, name='BatchHandlerMom1'
)
BatchHandlerMom1SF = BatchHandlerFactory(
    QueueMom1SF, Sampler, name='BatchHandlerMom1SF'
)
BatchHandlerMom2 = BatchHandlerFactory(
    QueueMom2, Sampler, name='BatchHandlerMom2'
)
BatchHandlerMom2Sep = BatchHandlerFactory(
    QueueMom2Sep, Sampler, name='BatchHandlerMom2Sep'
)
BatchHandlerMom2SF = BatchHandlerFactory(
    QueueMom2SF, Sampler, name='BatchHandlerMom2F'
)
BatchHandlerMom2SepSF = BatchHandlerFactory(
    QueueMom2SepSF, Sampler, name='BatchHandlerMom2SepSF'
)

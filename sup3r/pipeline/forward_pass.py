"""Sup3r forward pass handling module."""

import logging
import pprint
from concurrent.futures import as_completed
from datetime import datetime as dt
from typing import ClassVar

import numpy as np
from rex.utilities.execution import SpawnProcessPool
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.pipeline.strategy import ForwardPassChunk, ForwardPassStrategy
from sup3r.pipeline.utilities import get_model
from sup3r.preprocessing.utilities import (
    _mem_check,
    get_source_type,
    log_args,
    lowered,
)
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI
from sup3r.utilities.utilities import Timer
from sup3r.writers import OutputHandlerH5, OutputHandlerNC

logger = logging.getLogger(__name__)


class ForwardPass:
    """Class to run forward passes on all chunks provided by the given
    ForwardPassStrategy. The chunks provided by the strategy are all passed
    through the GAN generator to produce high resolution output.
    """

    OUTPUT_HANDLER_CLASS: ClassVar = {
        'nc': OutputHandlerNC,
        'h5': OutputHandlerH5,
    }

    @log_args
    def __init__(self, strategy, node_index=0):
        """Initialize ForwardPass with ForwardPassStrategy. The strategy
        provides the data chunks to run forward passes on

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node used to run forward pass
        """
        self.timer = Timer()
        self.strategy = strategy
        self.model = strategy.model
        self.node_index = node_index

        output_type = get_source_type(strategy.out_pattern)
        msg = f'Received bad output type {output_type}'
        assert output_type is None or output_type in list(
            self.OUTPUT_HANDLER_CLASS
        ), msg

    def get_input_chunk(self, chunk_index=0, mode='reflect'):
        """Get :class:`FowardPassChunk` instance for the given chunk index."""

        chunk = self.strategy.init_chunk(chunk_index)
        chunk.input_data, chunk.exo_data = self.pad_source_data(
            chunk.input_data, chunk.pad_width, chunk.exo_data, mode=mode
        )
        return chunk

    @property
    def meta(self):
        """Meta data dictionary for the forward pass run (to write to output
        files)."""
        meta_data = {
            'node_index': self.node_index,
            'creation_date': dt.now().strftime('%d/%m/%Y %H:%M:%S'),
            'model_meta': self.model.meta,
            'gan_params': self.model.model_params,
            'strategy_meta': self.strategy.meta,
        }
        return meta_data

    def _get_step_enhance(self, step):
        """Get enhancement factors for a given step and combine type.

        Parameters
        ----------
        step : dict
            Model step dictionary. e.g.
            ``{'model': 0, 'combine_type': 'input'}``

        Returns
        -------
        s_enhance : int
            Spatial enhancement factor for given step and combine type
        t_enhance : int
            Temporal enhancement factor for given step and combine type
        """
        combine_type = step['combine_type']
        model_step = step['model']
        msg = f'Received weird combine_type {combine_type} for step: {step}'
        assert combine_type in {'input', 'output', 'layer'}, msg
        if combine_type.lower() == 'input':
            if model_step == 0:
                s_enhance = 1
                t_enhance = 1
            else:
                s_enhance = np.prod(self.model.s_enhancements[:model_step])
                t_enhance = np.prod(self.model.t_enhancements[:model_step])

        else:
            s_enhance = np.prod(self.model.s_enhancements[: model_step + 1])
            t_enhance = np.prod(self.model.t_enhancements[: model_step + 1])
        return s_enhance, t_enhance

    def pad_source_data(self, input_data, pad_width, exo_data, mode='reflect'):
        """Pad the edges of the source data from the data handler.

        Parameters
        ----------
        input_data : Union[np.ndarray, da.core.Array]
            Source input data from data handler class, shape is:
            (spatial_1, spatial_2, temporal, features)
        pad_width : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        exo_data: dict
            Full exo_handler_kwargs dictionary with all feature entries. See
            :meth:`run_generator` for more information.
        mode : str
            Mode to use for padding. e.g. 'reflect'.

        Returns
        -------
        out : Union[np.ndarray, da.core.Array]
            Padded copy of source input data from data handler class, shape is:
            (spatial_1, spatial_2, temporal, features)
        exo_data : dict
            Same as input dictionary with s_enhance, t_enhance added to each
            step entry for all features

        """
        out = np.pad(input_data, (*pad_width, (0, 0)), mode=mode)
        logger.debug(
            'Padded input data shape from %s to %s using mode "%s" '
            'with padding argument: %s',
            input_data.shape,
            out.shape,
            mode,
            pad_width,
        )

        if exo_data is not None:
            for feature in exo_data:
                for i, step in enumerate(exo_data[feature]['steps']):
                    s_enhance, t_enhance = self._get_step_enhance(step)
                    exo_pad_width = (
                        *(
                            (en * pw[0], en * pw[1])
                            for en, pw in zip(
                                [s_enhance, s_enhance, t_enhance], pad_width
                            )
                        ),
                        (0, 0),
                    )
                    new_exo = step['data']
                    if len(new_exo.shape) == 3:
                        new_exo = np.expand_dims(new_exo, axis=2)
                        new_exo = np.repeat(
                            new_exo,
                            step['t_enhance'] * input_data.shape[2],
                            axis=2,
                        )
                    new_exo = np.pad(new_exo, exo_pad_width, mode=mode)
                    exo_data[feature]['steps'][i]['data'] = new_exo
                    logger.debug(
                        'Got exo data for feature: %s, model step: %d, '
                        's_enhance: %d, t_enhance: %d',
                        feature,
                        i,
                        s_enhance,
                        t_enhance,
                    )
        return out, exo_data

    @classmethod
    def run_generator(
        cls,
        data_chunk,
        hr_crop_slices,
        model,
        s_enhance=None,
        t_enhance=None,
        exo_data=None,
    ):
        """Run forward pass of the generator on smallest data chunk. Each chunk
        has a maximum shape given by self.strategy.fwp_chunk_shape.

        Parameters
        ----------
        data_chunk : ndarray
            Low res data chunk to go through generator
        hr_crop_slices : list
            List of slices for extracting cropped region of interest from
            output. Output can include an extra overlapping boundary to
            reduce chunking error. The cropping cuts off this padded region
            before stitching chunks.
        model : Sup3rGan
            A loaded Sup3rGan model (any model imported from sup3r.models).
        t_enhance : int
            Factor by which to enhance temporal resolution
        s_enhance : int
            Factor by which to enhance spatial resolution
        exo_data : dict | None
            Dictionary of exogenous feature data with entries describing
            whether features should be combined at input, a mid network layer,
            or with output. e.g.
            .. code-block:: JSON
                {
                    'topography': {'steps': [
                        {'combine_type': 'input', 'model': 0, 'data': ...},
                        {'combine_type': 'layer', 'model': 0, 'data': ...}]}
                }

        Returns
        -------
        ndarray
            High resolution data generated by GAN
        """
        temp = cls._reshape_data_chunk(model, data_chunk, exo_data)
        data_chunk, exo_data, i_lr_t, i_lr_s = temp
        try:
            fun = Timer()(model.generate, log=True)
            hi_res = fun(data_chunk, exogenous_data=exo_data)
        except Exception as e:
            msg = 'Forward pass failed on chunk with shape {}.'.format(
                data_chunk.shape
            )
            logger.exception(msg)
            raise RuntimeError(msg) from e
        if len(hi_res.shape) == 4:
            hi_res = np.expand_dims(np.transpose(hi_res, (1, 2, 0, 3)), axis=0)

        if (
            s_enhance is not None
            and hi_res.shape[1] != s_enhance * data_chunk.shape[i_lr_s]
        ):
            msg = (
                'The stated spatial enhancement of {}x did not match '
                'the low res / high res shapes of {} -> {}'.format(
                    s_enhance, data_chunk.shape, hi_res.shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if (
            t_enhance is not None
            and hi_res.shape[3] != t_enhance * data_chunk.shape[i_lr_t]
        ):
            msg = (
                'The stated temporal enhancement of {}x did not match '
                'the low res / high res shapes of {} -> {}'.format(
                    t_enhance, data_chunk.shape, hi_res.shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        return hi_res[0][hr_crop_slices]

    @staticmethod
    def _reshape_data_chunk(model, data_chunk, exo_data):
        """Reshape and transpose data chunk and exogenous data before being
        passed to the sup3r model.

        Note
        ----
        Exo data needs to be different shapes for 5D (Spatiotemporal) /
        4D (Spatial / Surface) models, and different models use different
        indices for spatial and temporal dimensions. These differences are
        handled here.

        Parameters
        ----------
        model : Sup3rGan
            Sup3rGan or similar sup3r model
        data_chunk : Union[np.ndarray, da.core.Array]
            Low resolution data for a single spatiotemporal chunk that is going
            to be passed to the model generate function.
        exo_data : dict | None
            Full exo_handler_kwargs dictionary with all feature entries. See
            :meth:`ForwardPass.run_generator` for more information.

        Returns
        -------
        data_chunk : Union[np.ndarray, da.core.Array]
            Same as input but reshaped to (temporal, spatial_1, spatial_2,
            features) if the model is a spatial-first model or
            (n_obs, spatial_1, spatial_2, temporal, features) if the
            model is spatiotemporal
        exo_data : dict | None
            Same reshaping procedure as for data_chunk applied to
            exo_data[feature]['steps'][...]['data']
        i_lr_t : int
            Axis index for the low-resolution temporal dimension
        i_lr_s : int
            Axis index for the low-resolution spatial_1 dimension
        """
        if exo_data is not None:
            for feature in exo_data:
                for i, entry in enumerate(exo_data[feature]['steps']):
                    models = getattr(model, 'models', [model])
                    msg = (
                        f'model index ({entry["model"]}) for exo step {i} '
                        'exceeds the number of model steps'
                    )
                    assert entry['model'] < len(models), msg
                    current_model = models[entry['model']]
                    if current_model.is_4d:
                        out = np.transpose(entry['data'], axes=(2, 0, 1, 3))
                    else:
                        out = np.expand_dims(entry['data'], axis=0)
                    exo_data[feature]['steps'][i]['data'] = np.asarray(out)

        if model.is_4d:
            i_lr_t = 0
            i_lr_s = 1
            data_chunk = np.transpose(data_chunk, axes=(2, 0, 1, 3))
        else:
            i_lr_t = 3
            i_lr_s = 1
            data_chunk = np.expand_dims(data_chunk, axis=0)

        return np.asarray(data_chunk), exo_data, i_lr_t, i_lr_s

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize ForwardPassStrategy and run ForwardPass
        on a single node based on an input config.

        Parameters
        ----------
        config : dict
            sup3r forward pass config with all necessary args and kwargs to
            initialize ForwardPassStrategy and run ForwardPass on a single
            node.
        """
        import_str = ''
        import_str += 'import time;\n'
        import_str += 'from gaps import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += (
            'from sup3r.pipeline.forward_pass '
            f'import ForwardPassStrategy, {cls.__name__}'
        )

        fwps_init_str = get_fun_call_str(ForwardPassStrategy, config)

        node_index = config['node_index']
        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = f'"sup3r", log_level="{log_level}"'
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (
            f"python -c '{import_str};\n"
            't0 = time.time();\n'
            f'logger = init_logger({log_arg_str});\n'
            f'strategy = {fwps_init_str};\n'
            f'{cls.__name__}.run(strategy, {node_index});\n'
            't_elap = time.time() - t0;\n'
        )

        pipeline_step = config.get('pipeline_step') or ModuleName.FORWARD_PASS
        cmd = BaseCLI.add_status_cmd(config, pipeline_step, cmd)
        cmd += ";'\n"

        return cmd.replace('\\', '/')

    @classmethod
    def _output_check(cls, out_data, allowed_const):
        """Check if forward pass output is constant or contains NaNs. This can
        happen when the chunk going through the forward pass is too big.
        This is due to a tensorflow padding bug, with the padding mode
        set to 'reflect'. With the currently preferred tensorflow
        version (2.15.1) this results in scrambled output rather than
        constant. https://github.com/tensorflow/tensorflow/issues/91027

        Parameters
        ----------
        out_data : ndarray
            Forward pass output corresponding to the given chunk index
        allowed_const : list | bool
            If your model is allowed to output a constant output, set this to
            True to allow any constant output or a list of allowed possible
            constant outputs. See :class:`ForwardPassStrategy` for more
            information on this argument.
        """
        failed = False
        if allowed_const is True:
            return failed
        if allowed_const is False:
            allowed_const = []
        elif not isinstance(allowed_const, (list, tuple)):
            allowed_const = [allowed_const]

        if np.isnan(out_data).any():
            msg = 'Forward pass output contains NaN values!'
            failed = True
            logger.error(msg)
            return failed

        if np.isinf(out_data).any():
            msg = 'Forward pass output contains Inf values!'
            failed = True
            logger.error(msg)
            return failed

        for i in range(out_data.shape[-1]):
            msg = f'All values are the same for feature channel {i}!'
            value0 = out_data[0, 0, 0, i]
            all_same = (value0 == out_data[..., i]).all()
            if all_same and value0 not in allowed_const:
                failed = True
                logger.error(msg)
                break
        return failed

    @classmethod
    def run(cls, strategy, node_index):
        """Runs forward passes on all spatiotemporal chunks for the given node
        index.

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node on which the forward passes for spatiotemporal chunks
            will be run.
        """
        if not strategy.node_finished(node_index):
            logger.info(
                'Starting forward pass on node %s with %s chunks using %s '
                'execution.',
                node_index,
                len(strategy.node_chunks[node_index]),
                'serial' if strategy.pass_workers == 1 else 'parallel',
            )
            if strategy.pass_workers == 1:
                cls._run_serial(strategy, node_index)
            else:
                cls._run_parallel(strategy, node_index)
            logger.info('Finished forward pass on node %s.', node_index)
            logger.debug(
                'Timing report:\n%s',
                pprint.pformat(strategy.timer.log, indent=2),
            )

    @classmethod
    def _run_serial(cls, strategy, node_index):
        """Runs forward passes, on all spatiotemporal chunks for the given node
        index, in serial.

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node on which the forward passes for spatiotemporal chunks
            will be run.
        """

        start = dt.now()
        logger.debug(f'Running forward passes on node {node_index} in serial.')
        fwp = cls(strategy, node_index=node_index)
        for i, chunk_index in enumerate(strategy.node_chunks[node_index]):
            now = dt.now()
            if not strategy.chunk_finished(chunk_index):
                chunk = fwp.get_input_chunk(chunk_index=chunk_index)
                failed, _ = cls.run_chunk(
                    chunk=chunk,
                    model_kwargs=strategy.model_kwargs,
                    model_class=strategy.model_class,
                    allowed_const=strategy.allowed_const,
                    model=fwp.model,
                    output_workers=strategy.output_workers,
                    invert_uv=strategy.invert_uv,
                    nn_fill=strategy.nn_fill,
                    overwrite=(not strategy.incremental),
                    meta=fwp.meta,
                )
                logger.debug(
                    'Finished forward pass on chunk_index='
                    f'{chunk_index} in {dt.now() - now}. {i + 1} of '
                    f'{len(strategy.node_chunks[node_index])} '
                    f'complete. {_mem_check()}.'
                )
                if failed:
                    msg = (
                        f'Forward pass for chunk_index {chunk_index} failed '
                        'with constant output or NaNs.'
                    )
                    raise MemoryError(msg)

        logger.info(
            'Finished forward passes on '
            f'{len(strategy.node_chunks[node_index])} chunks in '
            f'{dt.now() - start}'
        )

    @classmethod
    def _run_parallel(cls, strategy, node_index):
        """Runs forward passes, on all spatiotemporal chunks for the given node
        index, with data extraction and forward pass routines in parallel.

        Parameters
        ----------
        strategy : ForwardPassStrategy
            ForwardPassStrategy instance with information on data chunks to run
            forward passes on.
        node_index : int
            Index of node on which the forward passes for spatiotemporal chunks
            will be run.
        """

        logger.info(
            f'Running parallel forward passes on node {node_index}'
            f' with pass_workers={strategy.pass_workers}.'
        )

        futures = {}
        start = dt.now()
        pool_kws = {'max_workers': strategy.pass_workers, 'loggers': ['sup3r']}
        fwp = cls(strategy, node_index=node_index)
        with SpawnProcessPool(**pool_kws) as exe:
            now = dt.now()
            for _, chunk_index in enumerate(strategy.node_chunks[node_index]):
                if not strategy.chunk_finished(chunk_index):
                    chunk = fwp.get_input_chunk(chunk_index=chunk_index)
                    fut = exe.submit(
                        fwp.run_chunk,
                        chunk=chunk,
                        model_kwargs=strategy.model_kwargs,
                        model_class=strategy.model_class,
                        allowed_const=strategy.allowed_const,
                        output_workers=strategy.output_workers,
                        invert_uv=strategy.invert_uv,
                        nn_fill=strategy.nn_fill,
                        overwrite=(not strategy.incremental),
                        meta=fwp.meta,
                    )
                    futures[fut] = {
                        'chunk_index': chunk_index,
                        'start_time': dt.now(),
                    }

            logger.info(
                f'Started {len(futures)} forward pass runs in '
                f'{dt.now() - now}.'
            )

            try:
                for i, future in enumerate(as_completed(futures)):
                    failed, _ = future.result()
                    chunk_idx = futures[future]['chunk_index']
                    start_time = futures[future]['start_time']
                    if failed:
                        msg = (
                            f'Forward pass for chunk_index {chunk_idx} failed '
                            'with constant output or NaNs.'
                        )
                        raise MemoryError(msg)
                    msg = (
                        'Finished forward pass on chunk_index='
                        f'{chunk_idx} in {dt.now() - start_time}. '
                        f'{i + 1} of {len(futures)} complete. {_mem_check()}'
                    )
                    logger.debug(msg)
            except Exception as e:
                msg = (
                    'Error running forward pass on chunk_index='
                    f'{futures[future]["chunk_index"]}.'
                )
                logger.exception(msg)
                raise RuntimeError(msg) from e

        logger.info(
            'Finished asynchronous forward passes on '
            f'{len(strategy.node_chunks[node_index])} chunks in '
            f'{dt.now() - start}'
        )

    @classmethod
    def run_chunk(
        cls,
        chunk: ForwardPassChunk,
        model_kwargs,
        model_class,
        allowed_const,
        model=None,
        invert_uv=False,
        meta=None,
        nn_fill=True,
        output_workers=None,
        overwrite=False,
    ):
        """Run a forward pass on single spatiotemporal chunk.

        Parameters
        ----------
        chunk : :class:`FowardPassChunk`
            Struct with chunk data (including exo data if applicable) and
            chunk attributes (e.g. chunk specific slices, times, lat/lon, etc)
        model_kwargs : str | list
            Keyword arguments to send to `model_class.load(**model_kwargs)` to
            initialize the GAN. Typically this is just the string path to the
            model directory, but can be multiple models or arguments for more
            complex models.
        model_class : str
            Name of the sup3r model class for the GAN model to load. The
            default is the basic spatial / spatiotemporal Sup3rGan model. This
            will be loaded from sup3r.models
        allowed_const : list | bool
            If your model is allowed to output a constant output, set this to
            True to allow any constant output or a list of allowed possible
            constant outputs. See :class:`ForwardPassStrategy` for more
            information on this argument.
        model : Sup3rGan | None
            Optional preloaded model instance to reuse for the chunk. If
            ``None``, the model is loaded from ``model_class`` and
            ``model_kwargs``.
        invert_uv : bool
            Whether to convert uv to windspeed and winddirection for writing
            output. When this method is called during a pipeline forward pass
            the value is taken from the strategy.invert_uv attribute.
        nn_fill : bool
            Whether to fill data outside of accepted limits (e.g. relative
            humidity 0-100) with nearest neighbour or cap to limits. If true
            then any values outside of limits will be filled with the nearest
            neighbour value within limits. If false then values will be capped
            to limits. This is only used for writing output and does not affect
            the data passed through the generator.
        meta : dict | None
            Meta data to write to forward pass output file.
        output_workers : int | None
            Max number of workers to use for writing forward pass output.
        overwrite : bool
            Whether to overwrite existing output file or skip writing if file
            already exists. Default is False to avoid accidentally overwriting
            files.

        Returns
        -------
        failed : bool
            Whether the forward pass failed due to constant output.
        output_data : ndarray
            Array of high-resolution output from generator
        """

        msg = f'Running forward pass for chunk_index={chunk.index}.'
        logger.debug(msg)

        if model is None:
            model = get_model(model_class, model_kwargs)

        mask = np.isnan(chunk.input_data).any(axis=(0, 1, 2))
        feats = np.array(model.lr_features[: len(mask)])[mask]
        if np.any(mask):
            msg = (
                f'Input data for {feats} contains NaN values. Either '
                'use ``nan_method_kwargs`` to fill these on the fly or '
                'clean the data.'
            )
            logger.error(msg)
            raise RuntimeError(msg)

        mask = np.isinf(chunk.input_data).any(axis=(0, 1, 2))
        feats = np.array(model.lr_features[: len(mask)])[mask]
        if np.any(mask):
            msg = (
                f'Input data for {feats} contains Inf values. Cannot run '
                'forward pass.'
            )
            logger.error(msg)
            raise RuntimeError(msg)

        output_data = cls.run_generator(
            data_chunk=chunk.input_data,
            hr_crop_slices=chunk.hr_crop_slice,
            s_enhance=model.s_enhance,
            t_enhance=model.t_enhance,
            exo_data=chunk.exo_data,
            model=model,
        )

        failed = cls._output_check(output_data, allowed_const=allowed_const)

        if chunk.out_file is not None and not failed:
            logger.debug(f'Saving forward pass output to {chunk.out_file}.')
            output_type = get_source_type(chunk.out_file)
            cls.OUTPUT_HANDLER_CLASS[output_type]._write_output(
                data=output_data,
                features=lowered(model.hr_out_features),
                lat_lon=chunk.hr_lat_lon,
                row_inds=chunk.row_inds,
                col_inds=chunk.col_inds,
                times=chunk.hr_times,
                out_file=chunk.out_file,
                meta_data=meta,
                invert_uv=invert_uv,
                nn_fill=nn_fill,
                max_workers=output_workers,
                gids=chunk.gids,
                overwrite=overwrite,
            )
        return failed, output_data

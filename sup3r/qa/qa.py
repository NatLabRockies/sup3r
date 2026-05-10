"""sup3r QA module.

TODO: Good initial refactor but can do more cleaning here. Should use Loaders
and Sup3rX.unflatten() method (for H5) to make things more agnostic to dim
ordering.
"""

import logging
import os

import numpy as np
from rex import Resource
from rex.utilities.fun_utils import get_fun_call_str

from sup3r.bias.utilities import bias_correct_feature
from sup3r.preprocessing.derivers import Deriver
from sup3r.preprocessing.derivers.utilities import parse_feature
from sup3r.preprocessing.utilities import (
    get_input_handler_class,
    get_source_type,
    lowered,
)
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI
from sup3r.utilities.utilities import (
    OUTPUT_ATTRS,
    spatial_coarsening,
    temporal_coarsening,
    xr_open_mfdataset,
)
from sup3r.writers import RexOutputs

logger = logging.getLogger(__name__)


class Sup3rQa:
    """Class for doing QA on sup3r forward pass outputs.

    Note
    ----
    This only works if the sup3r forward pass output can be reshaped into a 2D
    raster dataset (e.g. no sparsifying of the meta data).
    """

    def __init__(
        self,
        source_file_paths,
        out_file_path,
        s_enhance,
        t_enhance,
        temporal_coarsening_method,
        features=None,
        source_features=None,
        output_names=None,
        input_handler_name=None,
        input_handler_kwargs=None,
        qa_fp=None,
        bias_correct_method=None,
        bias_correct_kwargs=None,
        save_sources=True,
    ):
        """
        Parameters
        ----------
        source_file_paths : list | str
            A list of low-resolution source files to extract raster data from.
            Each file must have the same number of timesteps. Can also pass a
            string with a unix-style file path which will be passed through
            glob.glob
        out_file_path : str
            A single sup3r-resolved output file (either .nc or .h5) with
            high-resolution data corresponding to the
            source_file_paths * s_enhance * t_enhance
        s_enhance : int
            Factor by which the Sup3rGan model will enhance the spatial
            dimensions of low resolution data
        t_enhance : int
            Factor by which the Sup3rGan model will enhance temporal dimension
            of low resolution data
        temporal_coarsening_method : str | list
            [subsample, average, total, min, max]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps. This can also be a list of method names corresponding
            to the list of features.
        features : str | list | None
            Explicit list of features to validate. Can be a single feature str,
            list of string feature names, or None for all features found in the
            out_file_path.
        source_features : str | list | None
            Optional feature names to retrieve from the source dataset if the
            source feature names are not the same as the sup3r output feature
            names. These will be used to derive the features to be validated.
            e.g. If model output is temperature_2m, and these were derived from
            temperature_min_2m (and max), then source features should be
            temperature_min_2m and temperature_max_2m while the model output
            temperature_2m is aggregated using min/max in the
            temporal_coarsening_method. Another example is features="ghi",
            source_features="rsds", where this is a simple alternative name
            lookup.
        output_names : str | list
            Optional output file dataset names corresponding to the features
            list input
        input_handler_name : str | None
            data handler class to use for input data. Provide a string name to
            match a class in sup3r.preprocessing.data_handlers. If None the
            correct handler will be guessed based on file type.
        input_handler_kwargs : dict
            Keyword arguments for `input_handler`. See :class:`Rasterizer`
            class for argument details.
        qa_fp : str | None
            Optional filepath to output QA file when you call Sup3rQa.run()
            (only .h5 is supported)
        bias_correct_method : str | None
            Optional bias correction function name that can be imported from
            the sup3r.bias.bias_transforms module. This will transform the
            source data according to some predefined bias correction
            transformation along with the bias_correct_kwargs. As the first
            argument, this method must receive a generic numpy array of data to
            be bias corrected
        bias_correct_kwargs : dict | None
            Optional namespace of kwargs to provide to bias_correct_method.
            If this is provided, it must be a dictionary where each key is a
            feature name and each value is a dictionary of kwargs to correct
            that feature. You can bias correct only certain input features by
            only including those feature names in this dict.
        save_sources : bool
            Flag to save re-coarsened synthetic data and true low-res data to
            qa_fp in addition to the error dataset
        """

        logger.info('Initializing Sup3rQa and retrieving source data...')

        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self._t_meth = temporal_coarsening_method
        self._out_fp = out_file_path
        self._features = (
            features if isinstance(features, (list, tuple)) else [features]
        )
        self._source_features = (
            source_features
            if isinstance(source_features, (list, tuple))
            else [source_features]
        )
        self._out_names = (
            output_names
            if isinstance(output_names, (list, tuple))
            else [output_names]
        )
        self.qa_fp = qa_fp
        self.save_sources = save_sources
        self.output_handler = (
            xr_open_mfdataset(self._out_fp)
            if self.output_type == 'nc'
            else Resource(self._out_fp)
        )

        self.bias_correct_method = bias_correct_method
        self.bias_correct_kwargs = (
            {}
            if bias_correct_kwargs is None
            else {k.lower(): v for k, v in bias_correct_kwargs.items()}
        )
        self.input_handler_kwargs = input_handler_kwargs or {}

        HandlerClass = get_input_handler_class(input_handler_name)
        self.input_handler = self.bias_correct_input_handler(
            HandlerClass(source_file_paths, **self.input_handler_kwargs)
        )
        self.meta = self.input_handler.data.meta
        self.time_index = self.input_handler.time_index

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise

    def close(self):
        """Close any open file handlers"""
        self.output_handler.close()

    @property
    def features(self):
        """Get a list of feature names from the output file, excluding meta and
        time index datasets

        Returns
        -------
        list
        """
        # all lower case
        ignore = ('meta', 'time_index', 'gids')

        if self._features is None or self._features == [None]:
            if self.output_type == 'nc':
                features = list(self.output_handler.data_vars)
            elif self.output_type == 'h5':
                features = self.output_handler.dsets
            features = [f for f in features if f.lower() not in ignore]

        elif isinstance(self._features, (list, tuple)):
            features = self._features

        return features

    @property
    def output_names(self):
        """Get a list of output dataset names corresponding to the features
        list
        """
        if self._out_names is None or self._out_names == [None]:
            return self.features
        return self._out_names

    @property
    def source_features(self):
        """Get a list of source dataset names corresponding to the input source
        data """
        if self._source_features is None or self._source_features == [None]:
            return self.features
        return self._source_features

    @property
    def output_type(self):
        """Get output data type

        Returns
        -------
        output_type
            e.g. 'nc' or 'h5'
        """
        ftype = get_source_type(self._out_fp)
        if ftype not in {'nc', 'h5'}:
            msg = 'Did not recognize output file type: {}'.format(self._out_fp)
            logger.error(msg)
            raise TypeError(msg)
        return ftype

    def bias_correct_input_handler(self, input_handler):
        """Apply bias correction to all source features which have bias
        correction data and return :class:`Deriver` instance to use for
        derivations of features to match output features.

        (1) Check if we need to derive any features included in the
        bias_correct_kwargs.
        (2) Derive these features using input_handler.resolve_feature, and
        update the stored data.
        (3) Apply bias correction to all the features in the
        bias_correct_kwargs
        (4) Derive the features required for validation from the bias corrected
        data and update the stored data
        (5) Return the updated input_handler, now a :class:`Deriver` object.
        """
        need_derive = list(
            set(lowered(self.bias_correct_kwargs))
            - set(input_handler.features)
        )
        msg = (
            f'Features {need_derive} need to be derived prior to bias '
            'correction, but the input_handler has no resolve_feature method. '
            'Request an appropriate input_handler with '
            'input_handler_name=DataHandlerName.'
        )
        assert len(need_derive) == 0 or hasattr(
            input_handler, 'resolve_feature'
        ), msg
        for f in need_derive:
            input_handler.data[f] = input_handler.resolve_feature(f)
        bc_feats = list(
            set(input_handler.features).intersection(
                set(lowered(self.bias_correct_kwargs.keys()))
            )
        )
        for f in bc_feats:
            input_handler.data[f] = bias_correct_feature(
                f,
                input_handler,
                self.bias_correct_method,
                self.bias_correct_kwargs,
            )

        return (
            input_handler
            if self.features in input_handler
            else Deriver(
                input_handler.data,
                features=self.features,
                FeatureRegistry=getattr(
                    input_handler, 'FEATURE_REGISTRY', None
                ),
            )
        )

    def get_dset_out(self, name):
        """Get an output dataset from the forward pass output file.

        TODO: Make this dim order agnostic. If we didnt have the h5 condition
        we could just do transpose('south_north', 'west_east', 'time')

        Parameters
        ----------
        name : str
            Name of the output dataset to retrieve. Must be found in the
            features property and the forward pass output file.

        Returns
        -------
        out : np.ndarray
            A copy of the high-resolution output data as a numpy
            array of shape (spatial_1, spatial_2, temporal)
        """

        logger.debug('Getting sup3r output dataset "{}"'.format(name))
        data = self.output_handler[name]
        if self.output_type == 'nc':
            data = data.values
        elif self.output_type == 'h5':
            shape = (
                len(self.input_handler.time_index) * self.t_enhance,
                int(self.input_handler.shape[0] * self.s_enhance),
                int(self.input_handler.shape[1] * self.s_enhance),
            )
            data = data.reshape(shape)
            # data always needs to be converted from (t, s1, s2) -> (s1, s2, t)
        data = np.transpose(data, axes=(1, 2, 0))

        return np.asarray(data)

    def coarsen_data(self, idf, feature, data):
        """Re-coarsen a high-resolution synthetic output dataset

        Parameters
        ----------
        idf : int
            Feature index
        feature : str
            Feature name
        data : Union[np.ndarray, da.core.Array]
            A copy of the high-resolution output data as a numpy
            array of shape (spatial_1, spatial_2, temporal)

        Returns
        -------
        data : Union[np.ndarray, da.core.Array]
            A spatiotemporally coarsened copy of the input dataset, still with
            shape (spatial_1, spatial_2, temporal)
        """
        t_meth = (
            self._t_meth
            if isinstance(self._t_meth, str)
            else self._t_meth[idf]
        )

        logger.info(
            f'Coarsening feature "{feature}" with {self.s_enhance}x '
            f'spatial averaging and "{t_meth}" {self.t_enhance}x '
            'temporal averaging'
        )

        data = spatial_coarsening(
            data, s_enhance=self.s_enhance, obs_axis=False
        )

        # t_coarse needs shape to be 5D: (obs, s1, s2, t, f)
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=4)
        data = temporal_coarsening(
            data, t_enhance=self.t_enhance, method=t_meth
        )
        data = data.squeeze(axis=(0, 4))

        return data

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize Sup3rQa and execute the Sup3rQa.run()
        method based on an input config

        Parameters
        ----------
        config : dict
            sup3r QA config with all necessary args and kwargs to
            initialize Sup3rQa and execute Sup3rQa.run()
        """
        import_str = 'import time;\n'
        import_str += 'from gaps import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += 'from sup3r.qa.qa import Sup3rQa'

        qa_init_str = get_fun_call_str(cls, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')

        log_arg_str = f'"sup3r", log_level="{log_level}"'
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (
            f"python -c '{import_str};\n"
            't0 = time.time();\n'
            f'logger = init_logger({log_arg_str});\n'
            f'qa = {qa_init_str};\n'
            'qa.run();\n'
            't_elap = time.time() - t0;\n'
        )

        pipeline_step = config.get('pipeline_step') or ModuleName.QA
        cmd = BaseCLI.add_status_cmd(config, pipeline_step, cmd)
        cmd += ";'\n"

        return cmd.replace('\\', '/')

    def export(self, qa_fp, data, dset_name, dset_suffix=''):
        """Export error dictionary to h5 file.

        Parameters
        ----------
        qa_fp : str | None
            Optional filepath to output QA file (only .h5 is supported)
        data : Union[np.ndarray, da.core.Array]
            An array with shape (space1, space2, time) that represents the
            re-coarsened synthetic data minus the source true low-res data, or
            another dataset of the same shape to be written to disk
        dset_name : str
            Base dataset name to save data to
        dset_suffix : str
            Optional suffix to append to dset_name with an underscore before
            saving.
        """

        if not os.path.exists(qa_fp):
            logger.info('Initializing qa output file: "{}"'.format(qa_fp))
            with RexOutputs(qa_fp, mode='w') as f:
                f.meta = self.input_handler.meta
                f.time_index = self.input_handler.time_index

        shape = (
            len(self.input_handler.time_index),
            len(self.input_handler.meta),
        )
        attrs = dict(OUTPUT_ATTRS.get(parse_feature(dset_name).basename, {}))

        # dont scale the re-coarsened data or diffs
        attrs['scale_factor'] = 1
        attrs['dtype'] = 'float32'

        if dset_suffix:
            dset_name = dset_name + '_' + dset_suffix

        logger.info('Adding dataset "{}" to output file.'.format(dset_name))

        # transpose and flatten to typical h5 (time, space) dimensions
        data = np.transpose(np.asarray(data), axes=(2, 0, 1)).reshape(shape)

        RexOutputs.add_dataset(
            qa_fp,
            dset_name,
            data,
            dtype=attrs['dtype'],
            chunks=attrs.get('chunks'),
            attrs=attrs,
        )

    def run(self):
        """Go through all datasets and get the error for the re-coarsened
        synthetic minus the true low-res source data.

        Returns
        -------
        errors : dict
            Dictionary of errors, where keys are the feature names, and each
            value is an array with shape (space1, space2, time) that represents
            the re-coarsened synthetic data minus the source true low-res data
        """

        errors = {}
        ziter = zip(self.features, self.source_features, self.output_names)
        for idf, (feature, source_feature, dset_out) in enumerate(ziter):
            logger.info(
                'Running QA on dataset {} of {} for feature "{}" '
                'with source feature name "{}"'.format(
                    idf + 1, len(self.features), feature, source_feature,
                )
            )
            data_syn = self.get_dset_out(feature)
            data_syn = self.coarsen_data(idf, feature, data_syn)
            data_true = self.input_handler[source_feature][...]

            if data_syn.shape != data_true.shape:
                msg = (
                    'Sup3rQa failed while trying to inspect the "{}" feature. '
                    'The source low-res data had shape {} while the '
                    're-coarsened synthetic data had shape {}.'.format(
                        feature, data_true.shape, data_syn.shape
                    )
                )
                logger.error(msg)
                raise RuntimeError(msg)

            feature_diff = data_syn - data_true
            errors[feature] = feature_diff

            if self.qa_fp is not None:
                self.export(self.qa_fp, feature_diff, dset_out, 'error')
                if self.save_sources:
                    self.export(self.qa_fp, data_syn, dset_out, 'synthetic')
                    self.export(self.qa_fp, data_true, dset_out, 'true')

        logger.info('Finished Sup3rQa run method.')

        return errors

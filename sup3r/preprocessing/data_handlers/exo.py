"""Exogenous data handler and related objects. The ExoDataHandler performs
exogenous data rasterization for one or more model steps for requested
features."""

import logging
import pathlib
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from sup3r.preprocessing.rasterizers import ExoRasterizer
from sup3r.preprocessing.utilities import _lowered, log_args

if TYPE_CHECKING:
    from sup3r.models import MultiStepGan, Sup3rGan

logger = logging.getLogger(__name__)


class SingleExoDataStep(dict):
    """Special dictionary class for exogenous_data step"""

    def __init__(self, feature, combine_type, model, data):
        """exogenous_data step dictionary for a given model step

        Parameters
        ----------
        feature : str
            Name of feature corresponding to ``data``.
        combine_type : str
            Specifies how the exogenous_data should be used for this step. e.g.
            "input", "layer", "output". For example, if tis equals "input" the
            ``data`` will be used as input to the forward pass for the model
            step given by ``model``
        model : int
            Specifies the model index which will use the ``data``. For example,
            if ``model`` == 1 then the ``data`` will be used according to
            ``combine_type`` in the 2nd model step in a MultiStepGan.
        data : Union[np.ndarray, da.core.Array]
            The data to be used for the given model step.
        """
        step = {'model': model, 'combine_type': combine_type, 'data': data}
        for k, v in step.items():
            self.__setitem__(k, v)
        self.feature = feature

    @property
    def shape(self):
        """Shape of data array for this model step."""
        return self['data'].shape


class ExoData(dict):
    """Special dictionary class for multiple exogenous_data steps

    TODO: Can we simplify this by relying more on xr.Dataset meta data instead
    of storing enhancement factors for each step? Seems like we could take the
    highest res data and coarsen based on s/t enhance, also.
    """

    def __init__(self, steps):
        """Combine multiple SingleExoDataStep objects

        Parameters
        ----------
        steps : dict
            Dictionary with feature keys each with entries describing whether
            features should be combined at input, a mid network layer, or with
            output. e.g.::

            \b
            {
                'topography': {
                    'steps': [
                        {'combine_type': 'input', 'model': 0, 'data': ...},
                        {'combine_type': 'layer', 'model': 0, 'data': ...}]
                    }
            }

            Each array in in 'data' key has 3D or 4D shape:
            (spatial_1, spatial_2, 1)
            (spatial_1, spatial_2, n_temporal, 1)

            If the 'combine_type' key is missing it is assumed to be 'layer',
            and if the 'steps' key is missing it is assumed to be a single
            step and will be converted to a list
        """  # noqa : D301
        if isinstance(steps, dict):
            for feat, entry in steps.items():
                msg = f'ExoData entry for {feat} must have a "steps" key.'
                assert 'steps' in entry, msg

                steps_list = entry['steps']
                for i, step in enumerate(steps_list):
                    msg = (
                        f'ExoData entry for {feat}, step #{i + 1}, must '
                        'have a "data" and "combine_type" key.'
                    )
                    assert 'data' in step and 'combine_type' in step, msg

            self.update(steps)

        else:
            msg = 'ExoData must be initialized with a dictionary of features.'
            logger.error(msg)
            raise ValueError(msg)

    def get_model_step_exo(self, model_step):
        """Get the exogenous data for the given model_step from the full list
        of steps

        Parameters
        ----------
        model_step : int
            Index of the model to get exogenous data for.

        Returns
        -------
        model_step_exo : dict
            Dictionary of features each with list of steps which match the
            given model_step
        """
        model_step_exo = {}
        for feature, entry in self.items():
            steps = [
                step for step in entry['steps'] if step['model'] == model_step
            ]
            if steps:
                model_step_exo[feature] = {'steps': steps}
        return ExoData(model_step_exo)

    @staticmethod
    def _get_bounded_steps(steps, min_step, max_step=None):
        """Get the steps within `steps` which have a model index between min
        and max step."""
        if max_step is not None:
            return [
                s
                for s in steps
                if (s['model'] < max_step and min_step <= s['model'])
            ]
        return [s for s in steps if min_step <= s['model']]

    def split(self, split_steps):
        """Split ``self`` into multiple ``ExoData`` objects based on
        ``split_steps``. The splits are done such that the steps in the ith
        entry of the returned list all have a
        ``model number < split_steps[i].``

        Note
        ----
        This is used for multi-step models to correctly distribute the set of
        all exo data steps to the appropriate models. For example,
        :class:`~sup3r.models.MultiStepGan` models with some
        temporal (spatial) steps followed by some spatial (temporal) steps. The
        temporal (spatial) models might take the first N exo data steps and
        then the spatial (temporal) models will take the remaining exo data
        steps.

        Parameters
        ----------
        split_steps : list
            Step index list to use for splitting. To split this into exo data
            for spatial models and temporal models split_steps should be
            ``[len(spatial_models)]``. If this is for a
            :class:`~sup3r.models.MultiStepGan` model with temporal steps
            first, ``split_steps`` should be ``[len(temporal_models)]``. If
            this is for a multi step model composed of more than two models
            (e.g. :class:`~sup3r.models.SolarMultiStepGan`) ``split_steps``
            should be ``[len(spatial_solar_models), len(spatial_solar_models) +
            len(spatial_wind_models)]``

        Returns
        -------
        split_list : list[ExoData]
            List of ``ExoData`` objects coming from the split of ``self``,
            according to ``split_steps``
        """
        split_dict = {i: {} for i in range(len(split_steps) + 1)}
        split_steps = [0, *split_steps] if split_steps[0] != 0 else split_steps
        for feature, entry in self.items():
            for i, min_step in enumerate(split_steps):
                max_step = (
                    None if min_step == split_steps[-1] else split_steps[i + 1]
                )
                steps_i = self._get_bounded_steps(
                    steps=entry['steps'], min_step=min_step, max_step=max_step
                )
                for s in steps_i:
                    s.update({'model': s['model'] - min_step})
                if steps_i:
                    split_dict[i][feature] = {'steps': steps_i}
        return [ExoData(split) for split in split_dict.values()]

    def get_combine_type_data(self, feature, combine_type, model_step=None):
        """Get exogenous data for given feature which is used according to the
        given combine_type (input/output/layer) for this model_step.

        Parameters
        ----------
        feature : str
            Name of exogenous feature to get data for
        combine_type : str
            Usage type for requested data. e.g input/output/layer
        model_step : int | None
            Model step the data will be used for. If this is not None then
            only steps with self[feature]['steps'][:]['model'] == model_step
            will be searched for data.

        Returns
        -------
        data : tf.Tensor | np.ndarray
            Exogenous data for given parameters
        """
        tmp = self[feature]
        if model_step is not None:
            tmp = {k: v for k, v in tmp.items() if v['model'] == model_step}
        combine_types = [step['combine_type'] for step in tmp['steps']]
        msg = (
            'Received exogenous_data without any combine_type '
            f'= "{combine_type}" steps.'
        )
        assert combine_type in combine_types, msg
        return tmp['steps'][combine_types.index(combine_type)]['data']

    @staticmethod
    def _get_enhanced_slices(lr_slices, step):
        """Get lr_slices enhanced by the ratio of exo_data_shape to
        input_data_shape. Used to slice exo data for each model step."""
        exo_slices = []
        for enhance, lr_slice in zip(
            [step['s_enhance'], step['s_enhance'], step['t_enhance']],
            lr_slices,
        ):
            exo_slc = slice(lr_slice.start * enhance, lr_slice.stop * enhance)
            exo_slices.append(exo_slc)
        return exo_slices

    def get_chunk(self, lr_slices):
        """Get the data for all model steps corresponding to the low res extent
        selected by `lr_slices`

        Parameters
        ----------
        lr_slices : list List of spatiotemporal slices which specify extent of
        the low-resolution input data.

        Returns
        -------
        exo_data : ExoData
           :class:`ExoData` object composed of multiple
           :class:`SingleExoDataStep` objects. This is the sliced exo data for
           the extent specified by `lr_slices`.
        """
        logger.debug('Getting exo data chunk for lr_slices=%s.', lr_slices)
        exo_chunk = {f: {'steps': []} for f in self}
        for feature in self:
            for step in self[feature]['steps']:
                exo_slices = self._get_enhanced_slices(
                    lr_slices=lr_slices, step=step
                )
                chunk_step = {}
                for k, v in step.items():
                    if k == 'data':
                        # last dimension is feature channel, so we use only the
                        # spatial slices if data is 2d and all slices otherwise
                        chunk_step[k] = v[
                            tuple(exo_slices)[: len(v.shape) - 1]
                        ]
                    else:
                        chunk_step[k] = v
                exo_chunk[feature]['steps'].append(chunk_step)
        return exo_chunk


class ExoDataHandler:
    """Class to rasterize exogenous features for multistep forward passes. e.g.
    Multiple topography arrays at different resolutions for multiple spatial
    enhancement steps.

    This takes a list of models and uses the different sets of models features
    to retrieve and rasterize exogenous data according to the requested target
    coordinate and grid shape, for each model step.
    """

    @log_args
    def __init__(
        self,
        feature: str,
        file_paths: Union[str, list, pathlib.Path],
        model: Optional[Union['Sup3rGan', 'MultiStepGan']] = None,
        steps: Optional[list] = None,
        **exo_rasterizer_kwargs,
    ):
        """
        Get list of steps with types of exogenous data needed for
        retrieval, initialize `self.data`, and update `self.data` for
        each model step with rasterized exo data.

        Parameters
        ----------
        feature : str
            Exogenous feature to extract from file_paths
        file_paths : str | list
            Filepaths(s) used to define the grid that the high-resolution
            exogenous data will be mapped onto. This can be either a single h5
            file or a list of netcdf files with identical grid. The string can
            be a unix-style file path which will be passed through glob.glob.
        model : Sup3rGan | MultiStepGan
            Model used to get exogenous data. If a ``MultiStepGan``
            ``lr_features``, ``hr_exo_features``, and
            ``hr_out_features`` will be checked for each model in
            ``model.models`` and exogenous data will be retrieved based
            on the resolution required for that type of feature. e.g. If
            a model has topography as a lr and hr_exo feature, and the
            model performs 5x spatial enhancement with an input
            resolution of 30km then topography at 30km and at 6km will
            be retrieved. Either this or list of steps needs to be
            provided.
        steps : list
            List of dictionaries containing info on which models to use
            for a given step index and what type of exo data the step
            requires. e.g.:: [{'model': 0, 'combine_type': 'input'},
             {'model': 0, 'combine_type': 'layer'}]
            Each step entry can also contain enhancement factors. e.g.::
            [{'model': 0, 'combine_type': 'input', 's_enhance': 1,
            't_enhance': 1},
             {'model': 0, 'combine_type': 'layer', 's_enhance': 3,
             't_enhance': 1}]
        exo_rasterizer_kwargs : dict
            Keyword arguments passed to the
            :class:`~sup3r.preprocessing.rasterizers.exo.BaseExoRasterizer`
            class. This is used to specify parameters for exogenous data
            rasterization such as the ``source_files`` for the exogenous data
            and the method of rasterization.
        """
        self.feature = feature
        self.file_paths = file_paths
        self.model = model
        self.steps = steps
        self.exo_rasterizer_kwargs = exo_rasterizer_kwargs
        self.models = getattr(self.model, 'models', [self.model])

        if self.steps is None:
            self.steps = self.get_exo_steps(self.feature, self.models)
        else:
            en_check = all('s_enhance' in v for v in self.steps)
            en_check = en_check and all('t_enhance' in v for v in self.steps)
            en_check = en_check or self.models is not None
            msg = (
                f'{self.__class__.__name__} needs s_enhance and t_enhance '
                'provided in each step in steps list or models'
            )
            assert en_check, msg
        self.s_enhancements, self.t_enhancements = self._get_all_enhancement()
        msg = (
            'Need to provide s_enhance and t_enhance for each model'
            'step. If the step is temporal only (spatial only) then '
            's_enhance = 1 (t_enhance = 1).'
        )
        self.data = self.get_all_step_data()

    @classmethod
    def get_exo_steps(cls, feature, models):
        """Get list of steps describing how to use exogenous data for the given
        feature in the list of given models. This checks the input and exo
        feature lists for each model step and adds that step if the given
        feature is found in the list."""
        steps = []
        for i, model in enumerate(models):
            is_sfc_model = model.__class__.__name__ == 'SurfaceSpatialMetModel'
            if feature.lower() in _lowered(model.lr_features) or is_sfc_model:
                steps.append({'model': i, 'combine_type': 'input'})
            if feature.lower() in _lowered(model.hr_exo_features):
                steps.append({'model': i, 'combine_type': 'layer'})
            if (
                feature.lower() in _lowered(model.hr_out_features)
                or is_sfc_model
            ):
                steps.append({'model': i, 'combine_type': 'output'})
        return steps

    def get_exo_rasterizer(self, s_enhance, t_enhance):
        """Get exo rasterizer instance for given enhancement factors"""
        return ExoRasterizer(
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            feature=self.feature,
            file_paths=self.file_paths,
            **self.exo_rasterizer_kwargs,
        )

    def get_single_step_data(self, s_enhance, t_enhance):
        """Get exo data for a single model step, with specific enhancement
        factors."""
        return self.get_exo_rasterizer(s_enhance, t_enhance).data

    @property
    def cache_files(self):
        """Get exo data cache file for all enhancement factors"""
        return [
            self.get_exo_rasterizer(s_en, t_en).cache_file
            for s_en, t_en in zip(self.s_enhancements, self.t_enhancements)
        ]

    def get_all_step_data(self):
        """Get exo data for each model step."""
        data = {self.feature: {'steps': []}}
        logger.debug(
            'Getting exo data for all steps with s_enhancements=%s and '
            't_enhancements=%s',
            self.s_enhancements,
            self.t_enhancements,
        )
        for i, (s_enhance, t_enhance) in enumerate(
            zip(self.s_enhancements, self.t_enhancements)
        ):
            step_data = self.get_single_step_data(
                s_enhance=s_enhance, t_enhance=t_enhance
            )
            step = SingleExoDataStep(
                self.feature,
                self.steps[i]['combine_type'],
                self.steps[i]['model'],
                data=step_data.as_array(),
            )
            step['s_enhance'] = s_enhance
            step['t_enhance'] = t_enhance
            data[self.feature]['steps'].append(step)
        shapes = [
            None if step is None else step.shape
            for step in data[self.feature]['steps']
        ]
        logger.debug(
            'Got exogenous_data of length %s with shapes: %s',
            len(data[self.feature]['steps']),
            shapes,
        )
        return data

    def _get_single_step_enhance(self, step):
        """Get enhancement factors for exogenous data extraction
        using exo_kwargs single model step. These factors are computed using
        stored enhance attributes of each model and the model step provided.
        If enhancement factors are already provided in step they are not
        overwritten.

        Parameters
        ----------
        step : dict
            Model step dictionary. e.g. {'model': 0, 'combine_type': 'input'}

        Returns
        -------
        updated_step : dict
            Same as input dictionary with s_enhance, t_enhance added
        """
        if all(key in step for key in ['s_enhance', 't_enhance']):
            return step

        mstep = step['model']
        combine_type = step.get('combine_type', None)
        msg = (
            f'Model index from exo_kwargs ({mstep} exceeds number '
            f'of model steps ({len(self.models)})'
        )
        assert len(self.models) > mstep, msg
        msg = (
            'Received exo_kwargs entry without valid combine_type '
            '(input/layer/output)'
        )
        assert combine_type.lower() in {'input', 'output', 'layer'}, msg
        if combine_type.lower() == 'input':
            if mstep == 0:
                s_enhance = 1
                t_enhance = 1
            else:
                s_enhance = int(np.prod(self.model.s_enhancements[:mstep]))
                t_enhance = int(np.prod(self.model.t_enhancements[:mstep]))

        else:
            s_enhance = int(np.prod(self.model.s_enhancements[: mstep + 1]))
            t_enhance = int(np.prod(self.model.t_enhancements[: mstep + 1]))
        step.update({'s_enhance': s_enhance, 't_enhance': t_enhance})
        return step

    def _get_all_enhancement(self):
        """Compute enhancement factors for all model steps.

        Returns
        -------
        s_enhancements: list
            List of s_enhance factors for all model steps
        t_enhancements: list
            List of t_enhance factors for all model steps
        """
        for i, step in enumerate(self.steps):
            out = self._get_single_step_enhance(step)
            self.steps[i] = out
        s_enhancements = [step['s_enhance'] for step in self.steps]
        t_enhancements = [step['t_enhance'] for step in self.steps]
        return s_enhancements, t_enhancements

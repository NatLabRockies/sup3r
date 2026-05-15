"""Sup3r multi step model frameworks

TODO: SolarMultiStepGan can be cleaned up a little with the output padding and
t_enhance argument moved to SolarCC.
"""

import logging
import os

import numpy as np
from gaps.config import load_config

# pylint: disable=cyclic-import
import sup3r.models
from sup3r.preprocessing.data_handlers import ExoData

from .base import Sup3rGan
from .interface import AbstractInterface

logger = logging.getLogger(__name__)


class MultiStepGan(AbstractInterface):
    """Multi-Step GAN, which is really just an abstraction layer on top of one
    or more Sup3rGan models that will perform their forward passes in
    serial."""

    def __init__(self, models):
        """
        Parameters
        ----------
        models : list | tuple
            An ordered list/tuple of one or more trained Sup3rGan models
        """
        self._models = tuple(models)

    def __len__(self):
        """Get number of model steps"""
        return len(self._models)

    @classmethod
    def load(cls, model_dirs, model_kwargs=None, verbose=True):
        """Load the GANs with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        model_dirs : list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
        model_kwargs : list | tuple
            An ordered list/tuple of one or more dictionaries containing kwargs
            for the corresponding model in model_dirs
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : MultiStepGan
            Returns a pretrained gan model that was previously saved to
            model_dirs
        """

        models = []
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]

        model_kwargs = model_kwargs or [{}] * len(model_dirs)
        if isinstance(model_kwargs, dict):
            model_kwargs = [model_kwargs]

        for model_dir, kwargs in zip(model_dirs, model_kwargs):
            fp_params = os.path.join(model_dir, 'model_params.json')
            assert os.path.exists(fp_params), f'Could not find: {fp_params}'
            params = load_config(fp_params)

            meta = params.get('meta', {'class': 'Sup3rGan'})
            class_name = meta.get('class', 'Sup3rGan')
            Sup3rClass = getattr(sup3r.models, class_name)
            models.append(
                Sup3rClass.load(model_dir, verbose=verbose, **kwargs)
            )

        return cls(models)

    @property
    def models(self):
        """Get an ordered tuple of the Sup3rGan models that are part of this
        MultiStepGan
        """
        return self._models

    @property
    def means(self):
        """Get the data normalization mean values. This is a tuple of means
        from all models.

        Returns
        -------
        tuple | np.ndarray
        """
        return tuple(model.means for model in self.models)

    @property
    def stdevs(self):
        """Get the data normalization standard deviation values. This is a
        tuple of stdevs from all models.

        Returns
        -------
        tuple | np.ndarray
        """
        return tuple(model.stdevs for model in self.models)

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducible results.

        Parameters
        ----------
        s : int
            Random seed
        """
        Sup3rGan.seed(s=s)

    def _transpose_model_input(self, model, hi_res):
        """Transpose input data according to mdel input dimensions.

        Note
        ----
        If hi_res.shape == 4, it is assumed that the dimensions have the
        ordering (n_obs, spatial_1, spatial_2, features)

        If hi_res.shape == 5, it is assumed that the dimensions have the
        ordering (1, spatial_1, spatial_2, temporal, features)

        Parameters
        ----------
        model : Sup3rGan
            A single step model with the attribute model.input_dims
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data transposed according
            to the number of model input dimensions
        """
        if model.is_5d and len(hi_res.shape) == 4:
            hi_res = np.transpose(hi_res, axes=(1, 2, 0, 3))[np.newaxis]
        elif model.is_4d and len(hi_res.shape) == 5:
            msg = (
                'Recieved 5D input data with shape '
                f'({hi_res.shape}) to a 4D model.'
            )
            assert hi_res.shape[0] == 1, msg
            hi_res = np.transpose(hi_res[0], axes=(2, 0, 1, 3))
        else:
            msg = (
                'Recieved input data with shape '
                f'{hi_res.shape} to a {model.input_dims}D model.'
            )
            assert model.input_dims == len(hi_res.shape), msg
        return hi_res

    def _match_model_input(self, model_step, hi_res, exo_data):
        """Match the output data of the previous model step to the input of
        the current model step. This is done to allow for model steps that
        only use a subset of the features from the previous model step."""
        if model_step > 0:
            current_model = self.models[model_step]
            previous_model = self.models[model_step - 1]
            output_feats = previous_model.hr_out_features
            input_feats = current_model.lr_features
            exo_data = exo_data or {}
            input_feats = [f for f in input_feats if f not in exo_data]
            if not set(input_feats).issubset(set(output_feats)):
                msg = (
                    'Model step {} input features {} do not match '
                    'previous model step {} output features {}'.format(
                        model_step, input_feats, model_step - 1, output_feats
                    )
                )
                logger.error(msg)
                raise ValueError(msg)
            lr_inds = [output_feats.index(fn) for fn in input_feats]
            hi_res = hi_res[..., lr_inds]
        return hi_res

    def generate(
        self, low_res, norm_in=True, un_norm_out=True, exogenous_data=None
    ):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : ExoData
           :class:`ExoData` object, which is a special dictionary containing
            exogenous data for each model step and info about how to use the
            data at each step.

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if isinstance(exogenous_data, dict) and not isinstance(
            exogenous_data, ExoData
        ):
            exogenous_data = ExoData(exogenous_data)

        hi_res = low_res.copy()
        for i, model in enumerate(self.models):
            i_norm_in = not (i == 0 and not norm_in)
            i_un_norm_out = not (i + 1 == len(self.models) and not un_norm_out)

            i_exo_data = (
                None
                if exogenous_data is None
                else exogenous_data.get_model_step_exo(i)
            )

            try:
                hi_res = self._transpose_model_input(model, hi_res)
                logger.debug(
                    'Data input to model #%s of %s has shape %s',
                    i + 1,
                    len(self.models),
                    hi_res.shape,
                )

                hi_res = self._match_model_input(i, hi_res, i_exo_data)

                hi_res = model.generate(
                    hi_res,
                    norm_in=i_norm_in,
                    un_norm_out=i_un_norm_out,
                    exogenous_data=i_exo_data,
                )
                logger.debug(
                    'Data output from model #%s of %s has shape %s',
                    i + 1,
                    len(self.models),
                    hi_res.shape,
                )
            except Exception as e:
                msg = (
                    'Could not run model #{} of {} "{}" '
                    'on tensor of shape {}'.format(
                        i + 1, len(self.models), model, hi_res.shape
                    )
                )
                logger.exception(msg)
                raise RuntimeError(msg) from e

        return hi_res

    @property
    def version_record(self):
        """Get a tuple of version records from all models

        Returns
        -------
        tuple
        """
        return tuple(model.version_record for model in self.models)

    @property
    def meta(self):
        """Get a tuple of meta data dictionaries for all models

        Returns
        -------
        tuple
        """
        return tuple(model.meta for model in self.models)

    @property
    def lr_features(self):
        """Get a list of low-resolution features input to the generative model.
        This includes low-resolution features that might be supplied
        exogenously at inference time but that were in the low-res batches
        during training"""
        return self.models[0].lr_features

    @property
    def hr_out_features(self):
        """Get the list of high-resolution output feature names that the
        generative model outputs."""
        return self.models[-1].hr_out_features

    @property
    def hr_exo_features(self):
        """Get list of high-resolution exogenous feature names the model uses.
        For the multi-step model, each entry in this list corresponds to one of
        the single-step models and is itself a list of hr_exo_features for that
        model.
        """
        return [model.hr_exo_features for model in self.models]

    @property
    def obs_features(self):
        """Get list of high-resolution observation feature names the model
        uses. For the multi-step model, each entry in this list corresponds to
        one of the single-step models and is itself a list of obs_features for
        that model.
        """
        return [model.obs_features for model in self.models]

    @property
    def model_params(self):
        """Get a tuple of model parameters for all models

        Returns
        -------
        tuple
        """
        return tuple(model.model_params for model in self.models)


class MultiStepSurfaceMetGan(MultiStepGan):
    """A two-step GAN where the first step is a spatial-only enhancement on a
    4D tensor of near-surface temperature and relative humidity data, and the
    second step is a (spatio)temporal enhancement on a 5D tensor.

    Note
    ----
    No inputs are needed for the first spatial-only surface meteorology
    model. The spatial enhancement is determined by the low and high res
    topography inputs in the exogenous_data kwargs in the
    MultiStepSurfaceMetGan.generate() method.

    The low res input to the spatial enhancement should be a 4D tensor of
    the shape (temporal, spatial_1, spatial_2, features) where temporal
    (usually the observation index) is a series of sequential timesteps that
    will be transposed to a 5D tensor of shape
    (1, spatial_1, spatial_2, temporal, features) tensor and then fed to the
    2nd-step (spatio)temporal GAN.
    """

    def generate(
        self, low_res, norm_in=True, un_norm_out=True, exogenous_data=None
    ):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution spatial input data, a 4D array of shape:
            (n_obs, spatial_1, spatial_2, n_features), Where the feature
            channel can include temperature_*m, relativehumidity_*m, and/or
            pressure_*m
        norm_in : bool
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : dict
            For the MultiStepSurfaceMetGan, this must be a nested dictionary
            with a main 'topography' key and two entries for
            exogenous_data['topography']['steps']. The first entry includes a
            2D (lat, lon) array of low-resolution surface elevation data in
            meters (must match spatial_1, spatial_2 from low_res), and the
            second entry includes a 2D (lat, lon) array of high-resolution
            surface elevation data in meters. e.g.
            .. code-block:: JSON
                {'topography': {
                    'steps': [
                        {'model': 0,
                         'combine_type':
                         'input',
                         'data': lr_topo},
                        {'model': 0,
                         'combine_type':
                         'output',
                         'data': hr_topo'}
                        ]
                    }
                }

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features), Where the
            feature channel can include temperature_*m, relativehumidity_*m,
            and/or pressure_*m
        """
        logger.debug(
            'Data input to the 1st step spatial-only enhancement has shape %s',
            low_res.shape,
        )

        msg = (
            'MultiStepSurfaceMetGan needs exogenous_data with two '
            'topography steps, for low and high res topography inputs.'
        )
        exo_check = (
            exogenous_data is not None
            and len(exogenous_data['topography']['steps']) == 2
        )
        assert exo_check, msg

        return super().generate(low_res, norm_in, un_norm_out, exogenous_data)

    @classmethod
    def load(
        cls,
        surface_model_class='SurfaceSpatialMetModel',
        temporal_model_class='MultiStepGan',
        surface_model_kwargs=None,
        temporal_model_kwargs=None,
        verbose=True,
    ):
        """Load the GANs with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        surface_model_class : str
            Name of surface model class to be retrieved from sup3r.models, this
            is typically "SurfaceSpatialMetModel"
        temporal_model_class : str
            Name of temporal model class to be retrieved from sup3r.models,
            this is typically "Sup3rGan"
        surface_model_kwargs : None | dict
            Optional additional kwargs to surface_model_class.load()
        temporal_model_kwargs : None | dict
            Optional additional kwargs to temporal_model_class.load()
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : MultiStepGan
            Returns a pretrained gan model that was previously saved to
            model_dirs
        """

        if surface_model_kwargs is None:
            surface_model_kwargs = {}

        if temporal_model_kwargs is None:
            temporal_model_kwargs = {}

        SpatialModelClass = getattr(sup3r.models, surface_model_class)
        s_models = SpatialModelClass.load(
            verbose=verbose, **surface_model_kwargs
        )

        TemporalModelClass = getattr(sup3r.models, temporal_model_class)
        t_models = TemporalModelClass.load(
            verbose=verbose, **temporal_model_kwargs
        )

        s_models = getattr(s_models, 'models', [s_models])
        t_models = getattr(t_models, 'models', [t_models])
        return cls([*s_models, *t_models])


class SolarMultiStepGan(MultiStepGan):
    """Special multi step model for solar clearsky ratio super resolution.

    This model takes in two parallel models for wind-only and solar-only
    spatial super resolutions, then combines them into a 3-channel
    high-spatial-resolution input (clearsky_ratio, u_200m, v_200m) for a solar
    temporal super resolution model.
    """

    def __init__(
        self,
        spatial_solar_models,
        spatial_wind_models,
        temporal_solar_models,
        t_enhance=None,
    ):
        """
        Parameters
        ----------
        spatial_solar_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more spatial
            super resolution steps in this composite MultiStepGan
            model that inputs and outputs clearsky_ratio
        spatial_wind_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more spatial
            super resolution steps in this composite MultiStepGan
            model that inputs and outputs wind u/v features and must include
            u_200m + v_200m as output features.
        temporal_solar_models : MultiStepGan
            A loaded MultiStepGan object representing the one or more
            (spatio)temporal super resolution steps in this composite
            SolarMultiStepGan model. This is the final step in the custom solar
            downscaling methodology.
        t_enhance : int | None
            Optional argument to fix or update the temporal enhancement of the
            model. This can be used to manipulate the output shape to match
            whatever padded shape the sup3r forward pass module expects. If
            this differs from the t_enhance value based on model layers the
            output will be padded so that the output shape matches low_res *
            t_enhance for the time dimension.
        """

        # Initializing parent without spatial solar models since this just
        # defines self.models. self.models is used to determine the aggregate
        # enhancement factors so including both spatial enhancement models
        # will results in an incorrect calculation.
        super().__init__(
            models=[
                *spatial_wind_models.models,
                *temporal_solar_models.models,
            ]
        )
        self._spatial_solar_models = spatial_solar_models
        self._spatial_wind_models = spatial_wind_models
        self._temporal_solar_models = temporal_solar_models
        self._t_enhance = t_enhance

        self.preflight()

        if self._t_enhance is not None:
            msg = (
                'Can only update t_enhance for a single temporal solar model.'
            )
            assert len(self.temporal_solar_models) == 1, msg
            model = self.temporal_solar_models.models[0]
            model.meta['t_enhance'] = self._t_enhance

    def preflight(self):
        """Run some preflight checks to make sure the loaded models can work
        together."""

        s_enh = self.spatial_solar_models.s_enhancements
        w_enh = self.spatial_wind_models.s_enhancements
        msg = (
            'Solar and wind spatial enhancements must be equivalent but '
            'received models that do spatial enhancements of '
            '{} (solar) and {} (wind)'.format(s_enh, w_enh)
        )
        assert np.prod(s_enh) == np.prod(w_enh), msg

        s_t_feat = self.spatial_solar_models.lr_features
        s_o_feat = self.spatial_solar_models.hr_out_features
        msg = (
            'Solar spatial enhancement models need to take '
            '"clearsky_ratio" as the only input and output feature but '
            'received models that need {} and output {}'.format(
                s_t_feat, s_o_feat
            )
        )
        assert s_t_feat == ['clearsky_ratio'], msg
        assert s_o_feat == ['clearsky_ratio'], msg

        temp_solar_feats = self.temporal_solar_models.lr_features
        msg = (
            'Input feature 0 for the temporal_solar_models should be '
            '"clearsky_ratio" but received: {}'.format(temp_solar_feats)
        )
        assert temp_solar_feats[0] == 'clearsky_ratio', msg

        spatial_out_features = (
            self.spatial_wind_models.hr_out_features
            + self.spatial_solar_models.hr_out_features
        )
        missing = [
            fn for fn in temp_solar_feats if fn not in spatial_out_features
        ]
        msg = (
            'Solar temporal model needs features {} that were not '
            'found in the solar + wind model output feature list {}'.format(
                missing, spatial_out_features
            )
        )
        assert not any(missing), msg

    @property
    def spatial_solar_models(self):
        """Get the MultiStepGan object for the spatial-only solar model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._spatial_solar_models

    @property
    def spatial_wind_models(self):
        """Get the MultiStepGan object for the spatial-only wind model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._spatial_wind_models

    @property
    def temporal_solar_models(self):
        """Get the MultiStepGan object for the (spatio)temporal model(s)

        Returns
        -------
        MultiStepGan
        """
        return self._temporal_solar_models

    @property
    def meta(self):
        """Get a tuple of meta data dictionaries for all models

        Returns
        -------
        tuple
        """
        return (
            self.spatial_solar_models.meta
            + self.spatial_wind_models.meta
            + self.temporal_solar_models.meta
        )

    @property
    def lr_features(self):
        """Get a list of low-resolution features input to the generative model.
        This includes low-resolution features that might be supplied
        exogenously at inference time but that were in the low-res batches
        during training"""
        return (
            self.spatial_solar_models.lr_features
            + self.spatial_wind_models.lr_features
        )

    @property
    def hr_out_features(self):
        """Get the list of output feature names that the last solar
        spatiotemporal generative model in this SolarMultiStepGan outputs."""
        return self.temporal_solar_models.hr_out_features

    @property
    def idf_wind(self):
        """Get an array of feature indices for the subset of features required
        for the spatial_wind_models. This excludes topography which is assumed
        to be provided as exogenous_data."""
        return np.array([
            self.lr_features.index(fn)
            for fn in self.spatial_wind_models.lr_features
            if fn != 'topography'
        ])

    @property
    def idf_wind_out(self):
        """Get an array of spatial_wind_models output feature indices that are
        required for input to the temporal_solar_models. Typically this is the
        indices of u_200m + v_200m from the output features of
        spatial_wind_models"""
        temporal_solar_features = self.temporal_solar_models.lr_features
        return np.array([
            self.spatial_wind_models.hr_out_features.index(fn)
            for fn in temporal_solar_features[1:]
        ])

    @property
    def idf_solar(self):
        """Get an array of feature indices for the subset of features required
        for the spatial_solar_models. This excludes topography which is assumed
        to be provided as exogenous_data."""
        return np.array([
            self.lr_features.index(fn)
            for fn in self.spatial_solar_models.lr_features
            if fn != 'topography'
        ])

    def generate(
        self, low_res, norm_in=True, un_norm_out=True, exogenous_data=None
    ):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data to the 1st step spatial GAN, which is a
            4D array of shape: (temporal, spatial_1, spatial_2, n_features).
            This should include all of the ``self.lr_features`` which is a
            concatenation of both the solar and wind spatial model features.
            The topography feature might be removed from this input and present
            in the exogenous_data input.
        norm_in : bool
            Flag to normalize low_res input data if the self.means,
            self.stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : ExoData | dict
           :class:`ExoData` object (or dict that will be cast to `ExoData`)
           with data arrays for each exogenous data step. Each array has 3D or
           4D shape:
            (spatial_1, spatial_2, n_features)
            (temporal, spatial_1, spatial_2, n_features)
            It's assumed that the spatial_solar_models do not require
            exogenous_data and only use clearsky_ratio.

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        """

        logger.debug(
            'Data input to the SolarMultiStepGan has shape %s which will be '
            'split up for solar- and wind-only features.',
            low_res.shape,
        )
        if isinstance(exogenous_data, dict) and not isinstance(
            exogenous_data, ExoData
        ):
            exogenous_data = ExoData(exogenous_data)

        if exogenous_data is not None:
            s_exo, t_exo = exogenous_data.split(
                split_steps=[len(self.spatial_wind_models)]
            )
        else:
            s_exo = t_exo = None

        try:
            hi_res_wind = self.spatial_wind_models.generate(
                low_res[..., self.idf_wind],
                norm_in=norm_in,
                un_norm_out=True,
                exogenous_data=s_exo,
            )
        except Exception as e:
            msg = (
                'Could not run the 1st step spatial-wind-only GAN on '
                'input shape {}'.format(low_res.shape)
            )
            logger.exception(msg)
            raise RuntimeError(msg) from e

        try:
            hi_res_solar = self.spatial_solar_models.generate(
                low_res[..., self.idf_solar], norm_in=norm_in, un_norm_out=True
            )
        except Exception as e:
            msg = (
                'Could not run the 1st step spatial-solar-only GAN on '
                'input shape {}'.format(low_res.shape)
            )
            logger.exception(msg)
            raise RuntimeError(msg) from e

        logger.debug(
            'Data output from the 1st step spatial enhancement has shape %s '
            '(solar) and shape %s (wind)',
            hi_res_solar.shape,
            hi_res_wind.shape,
        )

        hi_res = (hi_res_solar, hi_res_wind[..., self.idf_wind_out])
        hi_res = np.concatenate(hi_res, axis=3)

        logger.debug(
            'Data output from the concatenated solar + wind 1st step '
            'spatial-only enhancement has shape %s',
            hi_res.shape,
        )
        hi_res = np.transpose(hi_res, axes=(1, 2, 0, 3))
        hi_res = np.expand_dims(hi_res, axis=0)
        logger.debug(
            'Data from the concatenated solar + wind 1st step spatial-only '
            'enhancement has been reshaped to %s',
            hi_res.shape,
        )

        try:
            hi_res = self.temporal_solar_models.generate(
                hi_res,
                norm_in=True,
                un_norm_out=un_norm_out,
                exogenous_data=t_exo,
            )
        except Exception as e:
            msg = (
                'Could not run the 2nd step (spatio)temporal solar GAN on '
                'input shape {}'.format(low_res.shape)
            )
            logger.exception(msg)
            raise RuntimeError(msg) from e

        hi_res = self.temporal_pad(low_res, hi_res)

        logger.debug(
            'Final SolarMultiStepGan output has shape: %s', hi_res.shape
        )

        return hi_res

    def temporal_pad(self, low_res, hi_res, mode='reflect'):
        """Optionally add temporal padding to the 5D generated output array

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data to the 1st step spatial GAN, which is a
            4D array of shape: (temporal, spatial_1, spatial_2, n_features).
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
        mode : str
            Padding mode for np.pad()

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data output from the 2nd
            step (spatio)temporal GAN with a 5D array shape:
            (1, spatial_1, spatial_2, n_temporal, n_features)
            With the temporal axis padded with self._temporal_pad on either
            side.
        """
        t_shape = low_res.shape[0] * self.t_enhance
        t_pad = int((t_shape - hi_res.shape[-2]) / 2)
        pad_width = ((0, 0), (0, 0), (0, 0), (t_pad, t_pad), (0, 0))
        hi_res = np.pad(hi_res, pad_width, mode=mode)
        return hi_res

    @classmethod
    def load(
        cls,
        spatial_solar_model_dirs,
        spatial_wind_model_dirs,
        temporal_solar_model_dirs,
        t_enhance=None,
        verbose=True,
    ):
        """Load the GANs with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        spatial_solar_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only spatial solar models that input/output 4D
            tensors.
        spatial_wind_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only spatial wind models that input/output 4D
            tensors.
        temporal_solar_model_dirs : str | list | tuple
            An ordered list/tuple of one or more directories containing trained
            + saved Sup3rGan models created using the Sup3rGan.save() method.
            This must contain only (spatio)temporal solar models that
            input/output 5D tensors that are the concatenated output of the
            spatial_solar_models and the spatial_wind_models.
        t_enhance : int | None
            Optional argument to fix or update the temporal enhancement of the
            model. This can be used to manipulate the output shape to match
            whatever padded shape the sup3r forward pass module expects. If
            this differs from the t_enhance value based on model layers the
            output will be padded so that the output shape matches low_res *
            t_enhance for the time dimension.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : SolarMultiStepGan
            Returns a pretrained gan model that was previously saved to
            model_dirs
        """
        if isinstance(spatial_solar_model_dirs, str):
            spatial_solar_model_dirs = [spatial_solar_model_dirs]
        if isinstance(spatial_wind_model_dirs, str):
            spatial_wind_model_dirs = [spatial_wind_model_dirs]
        if isinstance(temporal_solar_model_dirs, str):
            temporal_solar_model_dirs = [temporal_solar_model_dirs]

        ssm = MultiStepGan.load(spatial_solar_model_dirs, verbose=verbose)
        swm = MultiStepGan.load(spatial_wind_model_dirs, verbose=verbose)
        tsm = MultiStepGan.load(temporal_solar_model_dirs, verbose=verbose)

        return cls(ssm, swm, tsm, t_enhance=t_enhance)

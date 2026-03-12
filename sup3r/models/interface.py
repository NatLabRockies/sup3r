"""Abstract class defining the required interface for Sup3r model subclasses"""

import json
import locale
import logging
import os
import re
from abc import ABC, abstractmethod
from warnings import warn

import numpy as np
from phygnn import CustomNetwork

from sup3r.preprocessing.data_handlers import ExoData
from sup3r.utilities import VERSION_RECORD
from sup3r.utilities.utilities import safe_cast

from .utilities import SUP3R_LAYERS

logger = logging.getLogger(__name__)


class AbstractInterface(ABC):
    """
    Abstract class to define the required interface for Sup3r model subclasses

    Note that this only sets the required interfaces for a GAN that can be
    loaded from disk and used to predict synthetic outputs. The interface for
    models that can be trained will be set in another class.
    """

    @classmethod
    @abstractmethod
    def load(cls, model_dir, verbose=True):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        model_dir
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : BaseModel
            Returns a pretrained gan model that was previously saved to
            model_dir
        """

    @abstractmethod
    def generate(
        self, low_res, norm_in=True, un_norm_out=True, exogenous_data=None
    ):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function."""

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducible results.

        Parameters
        ----------
        s : int
            Random seed
        """
        CustomNetwork.seed(s=s)

    @property
    def input_dims(self):
        """Get dimension of model generator input. This is usually 4D for
        spatial models and 5D for spatiotemporal models. This gives the input
        to the first step if the model is multi-step. Returns 5 for linear
        models.

        Returns
        -------
        int
        """
        if hasattr(self, '_gen'):
            return self._gen.layers[0].rank
        if hasattr(self, 'models'):
            return self.models[0].input_dims
        return 5

    @property
    def is_5d(self):
        """Check if model expects spatiotemporal input"""
        return self.input_dims == 5

    @property
    def is_4d(self):
        """Check if model expects spatial only input"""
        return self.input_dims == 4

    def get_s_enhance_from_layers(self):
        """Compute factor by which model will enhance spatial resolution from
        layer attributes. Used in model training during high res coarsening"""
        s_enhance = None
        if hasattr(self, '_gen'):
            s_enhancements = [
                getattr(layer, '_spatial_mult', 1)
                for layer in self._gen.layers
            ]
            s_enhance = int(np.prod(s_enhancements))
        return s_enhance

    def get_t_enhance_from_layers(self):
        """Compute factor by which model will enhance temporal resolution from
        layer attributes. Used in model training during high res coarsening"""
        t_enhance = None
        if hasattr(self, '_gen'):
            t_enhancements = [
                getattr(layer, '_temporal_mult', 1)
                for layer in self._gen.layers
            ]
            t_enhance = int(np.prod(t_enhancements))
        return t_enhance

    @property
    def s_enhance(self):
        """Factor by which model will enhance spatial resolution. Used in
        model training during high res coarsening and also in forward pass
        routine to determine shape of needed exogenous data"""
        models = getattr(self, 'models', [self])
        s_enhances = [m.meta.get('s_enhance', None) for m in models]
        s_enhance = (
            self.get_s_enhance_from_layers()
            if any(s is None for s in s_enhances)
            else int(np.prod(s_enhances))
        )
        if len(models) == 1 and isinstance(self.meta, dict):
            self.meta['s_enhance'] = s_enhance
        return s_enhance

    @property
    def t_enhance(self):
        """Factor by which model will enhance temporal resolution. Used in
        model training during high res coarsening and also in forward pass
        routine to determine shape of needed exogenous data"""
        models = getattr(self, 'models', [self])
        t_enhances = [m.meta.get('t_enhance', None) for m in models]
        t_enhance = (
            self.get_t_enhance_from_layers()
            if any(t is None for t in t_enhances)
            else int(np.prod(t_enhances))
        )
        if len(models) == 1 and isinstance(self.meta, dict):
            self.meta['t_enhance'] = t_enhance
        return t_enhance

    @property
    def s_enhancements(self):
        """List of spatial enhancement factors. In the case of a single step
        model this is just ``[self.s_enhance]``. This is used to determine
        shapes of needed exogenous data in forward pass routine"""
        if hasattr(self, 'models'):
            return [model.s_enhance for model in self.models]
        return [self.s_enhance]

    @property
    def t_enhancements(self):
        """List of temporal enhancement factors. In the case of a single step
        model this is just ``[self.t_enhance]``. This is used to determine
        shapes of needed exogenous data in forward pass routine"""
        if hasattr(self, 'models'):
            return [model.t_enhance for model in self.models]
        return [self.t_enhance]

    @property
    def input_resolution(self):
        """Resolution of input data. Given as a dictionary
        ``{'spatial': '...km', 'temporal': '...min'}``. The numbers are
        required to be integers in the units specified. The units are not
        strict as long as the resolution of the exogenous data, when extracting
        exogenous data, is specified in the same units."""
        input_resolution = self.meta.get('input_resolution', None)
        msg = 'model.input_resolution is None. This needs to be set.'
        assert input_resolution is not None, msg
        return input_resolution

    def _get_numerical_resolutions(self):
        """Get the input and output resolutions without units. e.g. for
        ``{"spatial": "30km", "temporal": "60min"}`` this returns
        ``{"spatial": 30, "temporal": 60}``"""
        ires_num = {
            k: int(re.search(r'\d+', v).group(0))
            for k, v in self.input_resolution.items()
        }
        enhancements = {'spatial': self.s_enhance, 'temporal': self.t_enhance}
        ores_num = {k: v // enhancements[k] for k, v in ires_num.items()}
        return ires_num, ores_num

    def _ensure_valid_input_resolution(self):
        """Ensure ehancement factors evenly divide input_resolution"""

        if self.input_resolution is None:
            return

        ires_num, ores_num = self._get_numerical_resolutions()
        s_enhance = self.meta['s_enhance']
        t_enhance = self.meta['t_enhance']
        check = (
            ires_num['temporal'] / ores_num['temporal'] == t_enhance
            and ires_num['spatial'] / ores_num['spatial'] == s_enhance
        )
        msg = (
            f'Enhancement factors (s_enhance={s_enhance}, '
            f't_enhance={t_enhance}) do not evenly divide '
            f'input resolution ({self.input_resolution})'
        )
        if not check:
            logger.error(msg)
            raise RuntimeError(msg)

    def _ensure_valid_enhancement_factors(self):
        """Ensure user provided enhancement factors are the same as those
        computed from layer attributes"""
        t_enhance = self.meta.get('t_enhance', None)
        s_enhance = self.meta.get('s_enhance', None)
        if s_enhance is None or t_enhance is None:
            return

        layer_se = self.get_s_enhance_from_layers()
        layer_te = self.get_t_enhance_from_layers()
        layer_se = layer_se if layer_se is not None else self.meta['s_enhance']
        layer_te = layer_te if layer_te is not None else self.meta['t_enhance']
        msg = (
            f'Enhancement factors computed from layer attributes '
            f'(s_enhance={layer_se}, t_enhance={layer_te}) '
            f'conflict with user provided values (s_enhance={s_enhance}, '
            f't_enhance={t_enhance})'
        )
        check = layer_se == s_enhance or layer_te == t_enhance
        if not check:
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def output_resolution(self):
        """Resolution of output data. Given as a dictionary
        {'spatial': '...km', 'temporal': '...min'}. This is computed from the
        input resolution and the enhancement factors."""
        output_res = self.meta.get('output_resolution', None)
        if self.input_resolution is not None and output_res is None:
            ires_num, ores_num = self._get_numerical_resolutions()
            output_res = {
                k: v.replace(str(ires_num[k]), str(ores_num[k]))
                for k, v in self.input_resolution.items()
            }
            self.meta['output_resolution'] = output_res
        return output_res

    def _combine_fwp_input(self, low_res, exogenous_data=None):
        """Combine exogenous_data at input resolution with low_res data prior
        to forward pass through generator

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        exogenous_data : dict | ExoData | None
            Special dictionary (class:`ExoData`) of exogenous feature data with
            entries describing whether features should be combined at input, a
            mid network layer, or with output. This doesn't have to include
            the 'model' key since this data is for a single step model.

        Returns
        -------
        low_res : np.ndarray
            Low-resolution input data combined with exogenous_data, usually a
            4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if exogenous_data is None:
            return low_res

        if (
            not isinstance(exogenous_data, ExoData)
            and exogenous_data is not None
        ):
            exogenous_data = ExoData(exogenous_data)

        fnum_diff = len(self.lr_features) - low_res.shape[-1]
        exo_feats = [] if fnum_diff <= 0 else self.lr_features[-fnum_diff:]
        msg = (
            f'Provided exogenous_data: {exogenous_data} is missing some '
            f'required features ({exo_feats})'
        )
        assert all(feature in exogenous_data for feature in exo_feats), msg
        if exogenous_data is not None and fnum_diff > 0:
            for feature in exo_feats:
                exo_input = exogenous_data.get_combine_type_data(
                    feature, 'input'
                )
                if exo_input is not None:
                    low_res = np.concatenate((low_res, exo_input), axis=-1)

        return low_res

    def _combine_fwp_output(self, hi_res, exogenous_data=None):
        """Combine exogenous_data at output resolution with generated hi_res
        data following forward pass output.

        Parameters
        ----------
        hi_res : np.ndarray
            High-resolution output data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        exogenous_data : dict | ExoData | None
            Special dictionary (class:`ExoData`) of exogenous feature data with
            entries describing whether features should be combined at input, a
            mid network layer, or with output. This doesn't have to include
            the 'model' key since this data is for a single step model.

        Returns
        -------
        hi_res : np.ndarray
            High-resolution output data combined with exogenous_data, usually a
            4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if exogenous_data is None:
            return hi_res

        if (
            not isinstance(exogenous_data, ExoData)
            and exogenous_data is not None
        ):
            exogenous_data = ExoData(exogenous_data)

        fnum_diff = len(self.hr_out_features) - hi_res.shape[-1]
        exo_feats = [] if fnum_diff <= 0 else self.hr_out_features[-fnum_diff:]
        msg = (
            'Provided exogenous_data is missing some required features '
            f'({exo_feats})'
        )
        assert all(feature in exogenous_data for feature in exo_feats), msg
        if exogenous_data is not None and fnum_diff > 0:
            for feature in exo_feats:
                exo_output = exogenous_data.get_combine_type_data(
                    feature, 'output'
                )
                if exo_output is not None:
                    hi_res = np.concatenate((hi_res, exo_output), axis=-1)
        return hi_res

    def get_layer_features(self):
        """Get the features that are input to mid-network layers. These are
        typically gapless high-resolution features like topography or sparse
        observations"""
        features = []
        if hasattr(self, '_gen'):
            for layer in self._gen.layers:
                if isinstance(layer, SUP3R_LAYERS):
                    feats = getattr(layer, 'features', [layer.name])
                    features.extend(feats)
        return features

    @property
    @abstractmethod
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

    @property
    def lr_features(self):
        """Get a list of low-resolution features input to the generative model.
        This includes low-resolution features that might be supplied
        exogenously at inference time but that were in the low-res batches
        during training"""
        return self.meta.get('lr_features', [])

    @property
    def hr_out_features(self):
        """Get the list of high-resolution output feature names that the
        generative model outputs."""
        return self.meta.get('hr_out_features', [])

    @property
    def obs_features(self):
        """Get list of exogenous observation feature names the model uses.
        These come from the names of the ``Sup3rObs..`` layers."""
        return self.meta.get('obs_features', [])

    @property
    def hr_exo_features(self):
        """Get list of gapless exogenous high-resolution feature names the
        model uses, like topography."""
        return self.meta.get('hr_exo_features', [])

    @property
    def hr_features(self):
        """Get the list of high-resolution feature names that are included in
        the high-resolution data during training. This includes both output
        and exogenous features.
        """
        return self.meta.get('hr_features', [])

    @property
    def smoothing(self):
        """Value of smoothing parameter used in gaussian filtering of coarsened
        high res data."""
        return self.meta.get('smoothing', None)

    @property
    def smoothed_features(self):
        """Get the list of smoothed input feature names that the generative
        model was trained on."""
        return self.meta.get('smoothed_features', [])

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        return {'meta': self.meta}

    @property
    def version_record(self):
        """A record of important versions that this model was built with.

        Returns
        -------
        dict
        """
        return VERSION_RECORD

    def set_model_params(self, **kwargs):
        """Set parameters used for training the model

        Parameters
        ----------
        kwargs : dict
            Keyword arguments including 'input_resolution',
            'lr_features', 'hr_exo_features', 'hr_out_features', 'hr_features',
            'obs_features', 'smoothed_features', 's_enhance', 't_enhance',
            'smoothing'
        """

        keys = (
            'input_resolution',
            'lr_features',
            'hr_exo_features',
            'hr_out_features',
            'hr_features',
            'obs_features',
            'smoothed_features',
            's_enhance',
            't_enhance',
            'smoothing',
        )
        keys = [k for k in keys if k in kwargs]
        if 'hr_out_features' in kwargs:
            self.meta['hr_out_features'] = kwargs['hr_out_features']

        for var in keys:
            val = self.meta.get(var, None)
            if val is None:
                self.meta[var] = kwargs[var]
            elif val != kwargs[var]:
                msg = (
                    'Model was previously trained with {var}={} but '
                    'received new {var}={}'.format(val, kwargs[var], var=var)
                )
                logger.warning(msg)
                warn(msg)

        self._ensure_valid_enhancement_factors()
        self._ensure_valid_input_resolution()

    def save_params(self, out_dir):
        """
        Parameters
        ----------
        out_dir : str
            Directory to save linear model params. This directory will be
            created if it does not already exist.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fp_params = os.path.join(out_dir, 'model_params.json')
        with open(
            fp_params, 'w', encoding=locale.getpreferredencoding(False)
        ) as f:
            params = self.model_params
            json.dump(params, f, sort_keys=True, indent=2, default=safe_cast)

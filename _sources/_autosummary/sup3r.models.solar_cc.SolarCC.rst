sup3r.models.solar\_cc.SolarCC
==============================

.. currentmodule:: sup3r.models.solar_cc

.. autoclass:: SolarCC
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~SolarCC.apply_grad_disc
      ~SolarCC.apply_grad_gen
      ~SolarCC.calc_loss
      ~SolarCC.calc_loss_disc
      ~SolarCC.calc_loss_gen_content
      ~SolarCC.calc_val_loss
      ~SolarCC.configure_multi_gpu
      ~SolarCC.dict_to_tensorboard
      ~SolarCC.discriminate
      ~SolarCC.early_stop
      ~SolarCC.finish_epoch
      ~SolarCC.generate
      ~SolarCC.get_hr_exo_input
      ~SolarCC.get_loss_fun
      ~SolarCC.get_optimizer_config
      ~SolarCC.get_optimizer_init_config
      ~SolarCC.get_optimizer_state
      ~SolarCC.get_single_grad_disc
      ~SolarCC.get_single_grad_gen
      ~SolarCC.get_weight_update_fraction
      ~SolarCC.init_optimizer
      ~SolarCC.init_weights
      ~SolarCC.load
      ~SolarCC.load_network
      ~SolarCC.load_saved_params
      ~SolarCC.log_loss_details
      ~SolarCC.norm_input
      ~SolarCC.profile_to_tensorboard
      ~SolarCC.run_exo_layer
      ~SolarCC.run_gradient_descent
      ~SolarCC.save
      ~SolarCC.save_params
      ~SolarCC.seed
      ~SolarCC.set_model_params
      ~SolarCC.set_norm_stats
      ~SolarCC.temporal_pad
      ~SolarCC.train
      ~SolarCC.un_norm_output
      ~SolarCC.update_adversarial_weights
      ~SolarCC.update_loss_details
      ~SolarCC.update_optimizer
      ~SolarCC.update_optimizer_disc
      ~SolarCC.update_optimizer_gen
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~SolarCC.DAYLIGHT_HOURS
      ~SolarCC.POINT_LOSS_HOURS
      ~SolarCC.STARTING_HOUR
      ~SolarCC.discriminator
      ~SolarCC.discriminator_weights
      ~SolarCC.generator
      ~SolarCC.generator_weights
      ~SolarCC.history
      ~SolarCC.hr_exo_features
      ~SolarCC.hr_features
      ~SolarCC.hr_out_features
      ~SolarCC.hr_out_features_ind
      ~SolarCC.input_dims
      ~SolarCC.input_resolution
      ~SolarCC.is_4d
      ~SolarCC.is_5d
      ~SolarCC.lr_features
      ~SolarCC.means
      ~SolarCC.meta
      ~SolarCC.model_params
      ~SolarCC.obs_features
      ~SolarCC.optimizer
      ~SolarCC.optimizer_disc
      ~SolarCC.output_resolution
      ~SolarCC.s_enhance
      ~SolarCC.s_enhancements
      ~SolarCC.smoothed_features
      ~SolarCC.smoothing
      ~SolarCC.stdevs
      ~SolarCC.strategy
      ~SolarCC.t_enhance
      ~SolarCC.t_enhancements
      ~SolarCC.total_batches
      ~SolarCC.version_record
      ~SolarCC.weights
   
   
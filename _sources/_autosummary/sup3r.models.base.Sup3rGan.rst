sup3r.models.base.Sup3rGan
==========================

.. currentmodule:: sup3r.models.base

.. autoclass:: Sup3rGan
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~Sup3rGan.apply_grad_disc
      ~Sup3rGan.apply_grad_gen
      ~Sup3rGan.calc_loss
      ~Sup3rGan.calc_loss_disc
      ~Sup3rGan.calc_loss_gen_content
      ~Sup3rGan.calc_val_loss
      ~Sup3rGan.configure_multi_gpu
      ~Sup3rGan.dict_to_tensorboard
      ~Sup3rGan.discriminate
      ~Sup3rGan.early_stop
      ~Sup3rGan.finish_epoch
      ~Sup3rGan.generate
      ~Sup3rGan.get_hr_exo_input
      ~Sup3rGan.get_loss_fun
      ~Sup3rGan.get_optimizer_config
      ~Sup3rGan.get_optimizer_init_config
      ~Sup3rGan.get_optimizer_state
      ~Sup3rGan.get_single_grad_disc
      ~Sup3rGan.get_single_grad_gen
      ~Sup3rGan.get_weight_update_fraction
      ~Sup3rGan.init_optimizer
      ~Sup3rGan.init_weights
      ~Sup3rGan.load
      ~Sup3rGan.load_network
      ~Sup3rGan.load_saved_params
      ~Sup3rGan.log_loss_details
      ~Sup3rGan.norm_input
      ~Sup3rGan.profile_to_tensorboard
      ~Sup3rGan.run_exo_layer
      ~Sup3rGan.run_gradient_descent
      ~Sup3rGan.save
      ~Sup3rGan.save_params
      ~Sup3rGan.seed
      ~Sup3rGan.set_model_params
      ~Sup3rGan.set_norm_stats
      ~Sup3rGan.train
      ~Sup3rGan.un_norm_output
      ~Sup3rGan.update_adversarial_weights
      ~Sup3rGan.update_loss_details
      ~Sup3rGan.update_optimizer
      ~Sup3rGan.update_optimizer_disc
      ~Sup3rGan.update_optimizer_gen
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Sup3rGan.discriminator
      ~Sup3rGan.discriminator_weights
      ~Sup3rGan.generator
      ~Sup3rGan.generator_weights
      ~Sup3rGan.history
      ~Sup3rGan.hr_exo_features
      ~Sup3rGan.hr_features
      ~Sup3rGan.hr_out_features
      ~Sup3rGan.hr_out_features_ind
      ~Sup3rGan.input_dims
      ~Sup3rGan.input_resolution
      ~Sup3rGan.is_4d
      ~Sup3rGan.is_5d
      ~Sup3rGan.lr_features
      ~Sup3rGan.means
      ~Sup3rGan.meta
      ~Sup3rGan.model_params
      ~Sup3rGan.obs_features
      ~Sup3rGan.optimizer
      ~Sup3rGan.optimizer_disc
      ~Sup3rGan.output_resolution
      ~Sup3rGan.s_enhance
      ~Sup3rGan.s_enhancements
      ~Sup3rGan.smoothed_features
      ~Sup3rGan.smoothing
      ~Sup3rGan.stdevs
      ~Sup3rGan.strategy
      ~Sup3rGan.t_enhance
      ~Sup3rGan.t_enhancements
      ~Sup3rGan.total_batches
      ~Sup3rGan.version_record
      ~Sup3rGan.weights
   
   
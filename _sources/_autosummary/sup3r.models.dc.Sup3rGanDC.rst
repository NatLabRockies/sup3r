sup3r.models.dc.Sup3rGanDC
==========================

.. currentmodule:: sup3r.models.dc

.. autoclass:: Sup3rGanDC
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~Sup3rGanDC.apply_grad_disc
      ~Sup3rGanDC.apply_grad_gen
      ~Sup3rGanDC.calc_loss
      ~Sup3rGanDC.calc_loss_disc
      ~Sup3rGanDC.calc_loss_gen_content
      ~Sup3rGanDC.calc_val_loss
      ~Sup3rGanDC.calc_val_loss_gen
      ~Sup3rGanDC.configure_multi_gpu
      ~Sup3rGanDC.dict_to_tensorboard
      ~Sup3rGanDC.discriminate
      ~Sup3rGanDC.early_stop
      ~Sup3rGanDC.finish_epoch
      ~Sup3rGanDC.generate
      ~Sup3rGanDC.get_hr_exo_input
      ~Sup3rGanDC.get_loss_fun
      ~Sup3rGanDC.get_optimizer_config
      ~Sup3rGanDC.get_optimizer_init_config
      ~Sup3rGanDC.get_optimizer_state
      ~Sup3rGanDC.get_single_grad_disc
      ~Sup3rGanDC.get_single_grad_gen
      ~Sup3rGanDC.get_weight_update_fraction
      ~Sup3rGanDC.init_optimizer
      ~Sup3rGanDC.init_weights
      ~Sup3rGanDC.load
      ~Sup3rGanDC.load_network
      ~Sup3rGanDC.load_saved_params
      ~Sup3rGanDC.log_loss_details
      ~Sup3rGanDC.norm_input
      ~Sup3rGanDC.profile_to_tensorboard
      ~Sup3rGanDC.run_exo_layer
      ~Sup3rGanDC.run_gradient_descent
      ~Sup3rGanDC.save
      ~Sup3rGanDC.save_params
      ~Sup3rGanDC.seed
      ~Sup3rGanDC.set_model_params
      ~Sup3rGanDC.set_norm_stats
      ~Sup3rGanDC.train
      ~Sup3rGanDC.un_norm_output
      ~Sup3rGanDC.update_adversarial_weights
      ~Sup3rGanDC.update_loss_details
      ~Sup3rGanDC.update_optimizer
      ~Sup3rGanDC.update_optimizer_disc
      ~Sup3rGanDC.update_optimizer_gen
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Sup3rGanDC.discriminator
      ~Sup3rGanDC.discriminator_weights
      ~Sup3rGanDC.generator
      ~Sup3rGanDC.generator_weights
      ~Sup3rGanDC.history
      ~Sup3rGanDC.hr_exo_features
      ~Sup3rGanDC.hr_features
      ~Sup3rGanDC.hr_out_features
      ~Sup3rGanDC.hr_out_features_ind
      ~Sup3rGanDC.input_dims
      ~Sup3rGanDC.input_resolution
      ~Sup3rGanDC.is_4d
      ~Sup3rGanDC.is_5d
      ~Sup3rGanDC.lr_features
      ~Sup3rGanDC.means
      ~Sup3rGanDC.meta
      ~Sup3rGanDC.model_params
      ~Sup3rGanDC.obs_features
      ~Sup3rGanDC.optimizer
      ~Sup3rGanDC.optimizer_disc
      ~Sup3rGanDC.output_resolution
      ~Sup3rGanDC.s_enhance
      ~Sup3rGanDC.s_enhancements
      ~Sup3rGanDC.smoothed_features
      ~Sup3rGanDC.smoothing
      ~Sup3rGanDC.stdevs
      ~Sup3rGanDC.strategy
      ~Sup3rGanDC.t_enhance
      ~Sup3rGanDC.t_enhancements
      ~Sup3rGanDC.total_batches
      ~Sup3rGanDC.version_record
      ~Sup3rGanDC.weights
   
   
sup3r.models.conditional.Sup3rCondMom
=====================================

.. currentmodule:: sup3r.models.conditional

.. autoclass:: Sup3rCondMom
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~Sup3rCondMom.apply_grad_disc
      ~Sup3rCondMom.apply_grad_gen
      ~Sup3rCondMom.calc_loss
      ~Sup3rCondMom.calc_loss_gen_content
      ~Sup3rCondMom.calc_val_loss
      ~Sup3rCondMom.configure_multi_gpu
      ~Sup3rCondMom.dict_to_tensorboard
      ~Sup3rCondMom.early_stop
      ~Sup3rCondMom.finish_epoch
      ~Sup3rCondMom.generate
      ~Sup3rCondMom.get_hr_exo_input
      ~Sup3rCondMom.get_loss_fun
      ~Sup3rCondMom.get_optimizer_config
      ~Sup3rCondMom.get_optimizer_init_config
      ~Sup3rCondMom.get_optimizer_state
      ~Sup3rCondMom.get_single_grad_disc
      ~Sup3rCondMom.get_single_grad_gen
      ~Sup3rCondMom.init_optimizer
      ~Sup3rCondMom.load
      ~Sup3rCondMom.load_network
      ~Sup3rCondMom.load_saved_params
      ~Sup3rCondMom.log_loss_details
      ~Sup3rCondMom.norm_input
      ~Sup3rCondMom.profile_to_tensorboard
      ~Sup3rCondMom.run_exo_layer
      ~Sup3rCondMom.run_gradient_descent
      ~Sup3rCondMom.save
      ~Sup3rCondMom.save_params
      ~Sup3rCondMom.seed
      ~Sup3rCondMom.set_model_params
      ~Sup3rCondMom.set_norm_stats
      ~Sup3rCondMom.train
      ~Sup3rCondMom.un_norm_output
      ~Sup3rCondMom.update_loss_details
      ~Sup3rCondMom.update_optimizer_gen
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Sup3rCondMom.generator
      ~Sup3rCondMom.generator_weights
      ~Sup3rCondMom.history
      ~Sup3rCondMom.hr_exo_features
      ~Sup3rCondMom.hr_features
      ~Sup3rCondMom.hr_out_features
      ~Sup3rCondMom.hr_out_features_ind
      ~Sup3rCondMom.input_dims
      ~Sup3rCondMom.input_resolution
      ~Sup3rCondMom.is_4d
      ~Sup3rCondMom.is_5d
      ~Sup3rCondMom.lr_features
      ~Sup3rCondMom.means
      ~Sup3rCondMom.meta
      ~Sup3rCondMom.model_params
      ~Sup3rCondMom.obs_features
      ~Sup3rCondMom.optimizer
      ~Sup3rCondMom.output_resolution
      ~Sup3rCondMom.s_enhance
      ~Sup3rCondMom.s_enhancements
      ~Sup3rCondMom.smoothed_features
      ~Sup3rCondMom.smoothing
      ~Sup3rCondMom.stdevs
      ~Sup3rCondMom.strategy
      ~Sup3rCondMom.t_enhance
      ~Sup3rCondMom.t_enhancements
      ~Sup3rCondMom.total_batches
      ~Sup3rCondMom.version_record
   
   
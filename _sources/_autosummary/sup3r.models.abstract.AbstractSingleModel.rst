sup3r.models.abstract.AbstractSingleModel
=========================================

.. currentmodule:: sup3r.models.abstract

.. autoclass:: AbstractSingleModel
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~AbstractSingleModel.apply_grad_disc
      ~AbstractSingleModel.apply_grad_gen
      ~AbstractSingleModel.calc_loss
      ~AbstractSingleModel.calc_loss_gen_content
      ~AbstractSingleModel.configure_multi_gpu
      ~AbstractSingleModel.dict_to_tensorboard
      ~AbstractSingleModel.early_stop
      ~AbstractSingleModel.finish_epoch
      ~AbstractSingleModel.generate
      ~AbstractSingleModel.get_hr_exo_input
      ~AbstractSingleModel.get_loss_fun
      ~AbstractSingleModel.get_optimizer_config
      ~AbstractSingleModel.get_optimizer_init_config
      ~AbstractSingleModel.get_optimizer_state
      ~AbstractSingleModel.get_single_grad_disc
      ~AbstractSingleModel.get_single_grad_gen
      ~AbstractSingleModel.init_optimizer
      ~AbstractSingleModel.load_network
      ~AbstractSingleModel.load_saved_params
      ~AbstractSingleModel.log_loss_details
      ~AbstractSingleModel.norm_input
      ~AbstractSingleModel.profile_to_tensorboard
      ~AbstractSingleModel.run_exo_layer
      ~AbstractSingleModel.run_gradient_descent
      ~AbstractSingleModel.save
      ~AbstractSingleModel.set_norm_stats
      ~AbstractSingleModel.un_norm_output
      ~AbstractSingleModel.update_loss_details
      ~AbstractSingleModel.update_optimizer_gen
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~AbstractSingleModel.generator
      ~AbstractSingleModel.generator_weights
      ~AbstractSingleModel.history
      ~AbstractSingleModel.means
      ~AbstractSingleModel.optimizer
      ~AbstractSingleModel.stdevs
      ~AbstractSingleModel.strategy
      ~AbstractSingleModel.total_batches
   
   
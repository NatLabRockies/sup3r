sup3r.pipeline.strategy.ForwardPassStrategy
===========================================

.. currentmodule:: sup3r.pipeline.strategy

.. autoclass:: ForwardPassStrategy
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~ForwardPassStrategy.chunk_finished
      ~ForwardPassStrategy.chunk_masked
      ~ForwardPassStrategy.get_chunk_indices
      ~ForwardPassStrategy.get_exo_cache_files
      ~ForwardPassStrategy.get_exo_kwargs
      ~ForwardPassStrategy.get_time_slices
      ~ForwardPassStrategy.init_chunk
      ~ForwardPassStrategy.init_input_handler
      ~ForwardPassStrategy.load_exo_data
      ~ForwardPassStrategy.node_finished
      ~ForwardPassStrategy.preflight
      ~ForwardPassStrategy.prep_chunk_data
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ForwardPassStrategy.allowed_const
      ~ForwardPassStrategy.bias_correct_kwargs
      ~ForwardPassStrategy.bias_correct_method
      ~ForwardPassStrategy.exo_handler_kwargs
      ~ForwardPassStrategy.fwp_chunk_shape
      ~ForwardPassStrategy.fwp_mask
      ~ForwardPassStrategy.grid_inds
      ~ForwardPassStrategy.head_node
      ~ForwardPassStrategy.hr_lat_lon
      ~ForwardPassStrategy.incremental
      ~ForwardPassStrategy.input_handler_kwargs
      ~ForwardPassStrategy.input_handler_name
      ~ForwardPassStrategy.invert_uv
      ~ForwardPassStrategy.max_nodes
      ~ForwardPassStrategy.meta
      ~ForwardPassStrategy.min_width
      ~ForwardPassStrategy.model_class
      ~ForwardPassStrategy.nn_fill
      ~ForwardPassStrategy.node_chunks
      ~ForwardPassStrategy.out_files
      ~ForwardPassStrategy.out_pattern
      ~ForwardPassStrategy.output_workers
      ~ForwardPassStrategy.pass_workers
      ~ForwardPassStrategy.redistribute_chunks
      ~ForwardPassStrategy.spatial_pad
      ~ForwardPassStrategy.temporal_pad
      ~ForwardPassStrategy.unmasked_chunks
      ~ForwardPassStrategy.use_cpu
      ~ForwardPassStrategy.file_paths
      ~ForwardPassStrategy.model_kwargs
   
   
import ml_collections

def get_SMIT_small_128_bias_True():
   '''
   A Small TransMorph Network
   '''
   config = ml_collections.ConfigDict()
   config.if_transskip = True
   config.if_convskip = True
   config.patch_size = 2
   config.in_chans = 1
   config.embed_dim = 48
   config.depths = (2, 2, 2, 2)
   config.num_heads = (3, 6, 12, 24)

   config.window_size = (4, 4, 4)

   config.mlp_ratio = 4
   config.pat_merg_rf = 4
   config.qkv_bias = True
   config.drop_rate = 0
   config.drop_path_rate = 0.3
   config.ape = False
   config.spe = False
   config.patch_norm = True
   config.use_checkpoint = False
   config.out_indices = (0, 1, 2, 3)
   config.reg_head_chan = 16
   config.img_size = (128, 128, 128)

   return config
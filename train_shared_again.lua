require 'nngraph'
require 'nn'
require 'hdf5'

require 'data.lua'
require 'util.lua'
require 'mymodels.lua'
require 'model_utils.lua'
require 'KLDCriterion.lua'
require 'latent_variable_sampler.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5',[[Path to the training *.hdf5 file 
                                               from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5',[[Path to validation *.hdf5 file 
                                                 from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as 
                         savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is 
                         the validation perplexity]])
cmd:option('-num_shards', 0, [[If the training data has been broken up into different shards, 
                             then training files are in this many partitions]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the
                                pretrained model.]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-z_size',500, [[Size of latent vector]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-hid_rec_size', 100, [[The hidden size of the recognition model]])
cmd:option('-use_chars_enc', 0, [[If = 1, use character on the encoder 
                                side (instead of word embeddings]])
cmd:option('-use_chars_dec', 0, [[If = 1, use character on the decoder 
                                side (instead of word embeddings]])
cmd:option('-reverse_src', 0, [[If = 1, reverse the source sequence. The original 
                              sequence-to-sequence paper found that this was crucial to 
                              achieving good performance, but with attention models this
                              does not seem necessary. Recommend leaving it to 0]])
cmd:option('-init_dec', 0, [[Initialize the hidden/cell state of the decoder at time 
                           0 to be the last hidden/cell state of the encoder. If 0, 
                           the initial states of the decoder are set to zero vectors]])
cmd:option('-input_feed', 1, [[If = 1, feed the context vector at each time step as additional
                             input (vica concatenation with the word embeddings) to the decoder]])
cmd:option('-multi_attn', 0, [[If > 0, then use a another attention layer on this layer of 
                           the decoder. For example, if num_layers = 3 and `multi_attn = 2`, 
                           then the model will do an attention over the source sequence
                           on the second layer (and use that as input to the third layer) and 
                           the penultimate layer]])
cmd:option('-res_net', 0, [[Use residual connections between LSTM stacks whereby the input to 
                          the l-th LSTM layer if the hidden state of the l-1-th LSTM layer 
                          added with the l-2th LSTM layer. We didn't find this to help in our 
                          experiments]])

cmd:text("")
cmd:text("Below options only apply if using the character model.")
cmd:text("")

-- char-cnn model specs (if use_chars == 1)
cmd:option('-char_vec_size', 25, [[Size of the character embeddings]])
cmd:option('-kernel_width', 6, [[Size (i.e. width) of the convolutional filter]])
cmd:option('-num_kernels', 1000, [[Number of convolutional filters (feature maps). So the
                                 representation from characters will have this many dimensions]])
cmd:option('-num_highway_layers', 2, [[Number of highway layers in the character model]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- optimization
cmd:option('-epochs', 1000, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support
                               (-param_init, param_init)]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If AdaGrad is used, then this is the
                                  global learning rate.]])
cmd:option('-adagrad', 0, [[Use AdaGrad instead of vanilla SGD.]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this, renormalize it
                                to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.2, [[Dropout probability. 
                            Dropout is applied between vertical LSTM stacks.]])
cmd:option('-lr_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                      on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 9, [[Start decay after this epoch]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load 
                                      pretrained word embeddings (hdf5 file) on the encoder side. 
                                      See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load 
                                      pretrained word embeddings (hdf5 file) on the decoder side. 
                                      See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])
cmd:option('-max_batch_l', '', [[If blank, then it will infer the max batch size from validation 
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])
cmd:text("")
cmd:text("**Other options**")
cmd:text("")


cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
                       on the source side. We've found this to make minimal difference]])
-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-gpuid2', -1, [[If this is >= 0, then the model will use two GPUs whereby the encoder
                           is on the first GPU and the decoder is on the second GPU. 
                           This will allow you to train with bigger batches/models.]])
cmd:option('-cudnn', 0, [[Whether to use cudnn or not for convolutions (for the character model).
                         cudnn has much faster convolutions so this is highly recommended 
                         if using the character model]])
-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

opt = cmd:parse(arg)
print(opt)
torch.manualSeed(opt.seed)

function zero_table(t)
   for i = 1, #t do
      t[i]:zero()
   end
end

function train(train_data, valid_data)

   local timer = torch.Timer()
   local num_params = 0
   local start_decay = 0
   params, grad_params = {}, {}
   opt.train_perf = {}
   opt.val_perf = {}
   
   for i = 1, #layers do
     
      local p, gp = layers[i]:getParameters()
      if opt.train_from:len() == 0 then
	 p:uniform(-opt.param_init, opt.param_init)
      end
      if i==1 then
            print(p:size())
            print(p:norm())
      end

      num_params = num_params + p:size(1)
      params[i] = p
      grad_params[i] = gp
   end

   if opt.pre_word_vecs_enc:len() > 0 then   
      local f = hdf5.open(opt.pre_word_vecs_enc)     
      local pre_word_vecs = f:read('word_vecs'):all()
      for i = 1, pre_word_vecs:size(1) do
	 word_vec_layers[1].weight[i]:copy(pre_word_vecs[i])
      end      
   end
   if opt.pre_word_vecs_dec:len() > 0 then      
      local f = hdf5.open(opt.pre_word_vecs_dec)     
      local pre_word_vecs = f:read('word_vecs'):all()
      for i = 1, pre_word_vecs:size(1) do
	 word_vec_layers[2].weight[i]:copy(pre_word_vecs[i])
      end      
   end
   
   print("Number of parameters: " .. num_params)
   
   word_vec_layers[1].weight[1]:zero() -- word vecs for enc source            
   word_vec_layers[2].weight[1]:zero() -- word vecs for enc target
   word_vec_layers[3].weight[1]:zero() -- word vecs for dec
   
   -- prototypes for gradients so there is no need to clone
   local encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
   local sampler_proto = torch.zeros(opt.max_batch_l, opt.z_size)
   context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
   
   -- clone encoder/decoder up to max source/max(target, source) length   
   decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ)
   encoder_clones_source = clone_many_times(encoder_source, opt.max_sent_l_src)
   encoder_clones_target = clone_many_times(encoder_target, opt.max_sent_l_targ)

   -- tie weights in different 
   for i = 1, opt.max_sent_l_src do
      if encoder_clones_source[i].apply then
	 encoder_clones_source[i]:apply(function(m) m:setReuse() end)
      end
   end

   for i = 1, opt.max_sent_l_targ do
      if encoder_clones_target[i].apply then
         encoder_clones_target[i]:apply(function(m) m:setReuse() end)
      end
   end

   for i = 1, opt.max_sent_l_targ do
      if decoder_clones[i].apply then
	 decoder_clones[i]:apply(function(m) m:setReuse() end)
      end
   end   

   local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
   if opt.gpuid >= 0 then
      h_init = h_init:cuda()      
      cutorch.setDevice(opt.gpuid)
      context_proto = context_proto:cuda()
      encoder_grad_proto = encoder_grad_proto:cuda()
      sampler_proto = sampler_proto:cuda()
   end

   init_fwd_enc = {}
   init_bwd_enc = {}
   init_fwd_dec = {}
   init_bwd_dec = {}
   if opt.input_feed == 1 then
      table.insert(init_fwd_dec, h_init:clone())
   end
   table.insert(init_bwd_dec, h_init:clone())
   
   for L = 1, opt.num_layers do
      table.insert(init_fwd_enc, h_init:clone())
      table.insert(init_fwd_enc, h_init:clone())
      table.insert(init_bwd_enc, h_init:clone())
      table.insert(init_bwd_enc, h_init:clone())
      table.insert(init_fwd_dec, h_init:clone()) -- memory cell
      table.insert(init_fwd_dec, h_init:clone()) -- hidden state
      table.insert(init_bwd_dec, h_init:clone())
      table.insert(init_bwd_dec, h_init:clone())      
   end      

   dec_offset = 3 -- offset depends on input feeding
   if opt.input_feed == 1 then
      dec_offset = dec_offset + 1
   end
   
   function reset_state(state, batch_l, t)
      if t == nil then
	 local u = {}
	 for i = 1, #state do
	    state[i]:zero()
	    table.insert(u, state[i][{{1, batch_l}}])
	 end
	 return u
      else
	 local u = {[t] = {}}
	 for i = 1, #state do
	    state[i]:zero()
	    table.insert(u[t], state[i][{{1, batch_l}}])
	 end
	 return u
      end      
   end

   -- clean layer before saving to make the model smaller
   function clean_layer(layer)
      if opt.gpuid >= 0 then
	 layer.output = torch.CudaTensor()
	 layer.gradInput = torch.CudaTensor()
      else
	 layer.output = torch.DoubleTensor()
	 layer.gradInput = torch.DoubleTensor()
      end
      if layer.modules then
	 for i, mod in ipairs(layer.modules) do
	    clean_layer(mod)
	 end
      elseif torch.type(self) == "nn.gModule" then
	 layer:apply(clean_layer)
      end      
   end

   -- decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
   function decay_lr(epoch)
      print(opt.val_perf)
      if epoch >= opt.start_decay_at then
	 start_decay = 1
      end
      
      if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
	 local curr_ppl = opt.val_perf[#opt.val_perf]
	 local prev_ppl = opt.val_perf[#opt.val_perf-1]
	 if curr_ppl > prev_ppl then
	    start_decay = 1
	 end
      end
      if start_decay == 1 then
	 opt.learning_rate = opt.learning_rate * opt.lr_decay
      end
   end   

   function train_batch(data, epoch)
      local train_nonzeros = 0
      local train_kldloss = 0	      
      local train_lossMLE = 0
      local batch_order = torch.randperm(data.length) -- shuffle mini batch order     
      local start_time = timer:time().real
      local num_words_target = 0
      local num_words_source = 0

      for i = 1, data:size() do
	 zero_table(grad_params, 'zero')
	 local d
         if epoch <= opt.curriculum then
	    d = data[i]
	 else
	    d = data[batch_order[i]]
	 end
         local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
	 local batch_l, target_l, source_l = d[5], d[6], d[7]

	 local encoder_grads_source = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]:clone()
	 local encoder_grads_target = encoder_grad_proto[{{1, batch_l}, {1, target_l}}]:clone()

        
	 local dz = sampler_proto[{{1,batch_l}, {1, opt.z_size}}]:clone() 
	 
	 local rnn_state_enc_source = reset_state(init_fwd_enc, batch_l, 0)
	 local rnn_state_enc_target = reset_state(init_fwd_enc, batch_l, 0)
	 
	 local context_source = context_proto[{{1, batch_l}, {1, source_l}}]:clone()
	 local context_target = context_proto[{{1, batch_l}, {1, target_l}}]:clone()

	 if opt.gpuid >= 0 then
	    cutorch.setDevice(opt.gpuid)
	 end	 

	 -- forward prop encoder source
	 for t = 1, source_l do
	    encoder_clones_source[t]:training()
	    local encoder_input = {source[t], table.unpack(rnn_state_enc_source[t-1])}
	    local out = encoder_clones_source[t]:forward(encoder_input)
	    rnn_state_enc_source[t] = out
	    context_source[{{},t}]:copy(out[#out])
	 end
         
	 -- forward prop encoder target
	 for t = 1, target_l do
	    encoder_clones_target[t]:training()
	    local encoder_input = {target[t], table.unpack(rnn_state_enc_target[t-1])}
	    local out = encoder_clones_target[t]:forward(encoder_input)
	    rnn_state_enc_target[t] = out
	    context_target[{{},t}]:copy(out[#out])
	 end
	

         -- use x and y to predict m and sigma from recognition model
	 local stats_phi = recognition_model:forward({context_source[{{},source_l}], context_target[{{},target_l}]})
	 local mu_phi = stats_phi[1]
	 local logsigma_phi = stats_phi[2]

         
	 -- calculate z (NOTE ONLY ONE SAMPLE)
	 local z = sampler:forward({mu_phi, logsigma_phi})
	 -- use x to predict m and s from prior model
	 local stats_omega = prior_model:forward(context_source[{{},source_l}])
	 local mu_omega = stats_omega[1]
	 local logsigma_omega = stats_omega[2]
	 
	 -- forward prop decoder
	 local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)

	 local preds = {}
	 local decoder_input
	 for t = 1, target_l do
	    decoder_clones[t]:training()
	    local decoder_input
	    decoder_input = {target[t], z, table.unpack(rnn_state_dec[t-1])}
	    local out = decoder_clones[t]:forward(decoder_input)
	    local next_state = {}
	    table.insert(preds, out[#out])
	    if opt.input_feed == 1 then
	       table.insert(next_state, out[#out])
	    end
	    for j = 1, #out-1 do
	       table.insert(next_state, out[j])
	    end
	    rnn_state_dec[t] = next_state
	 end
	
	 --zero grads
	 encoder_grads_source:zero()
	 encoder_grads_target:zero()
	 dz:zero()

	 -- backward prop decoder
	 local drnn_state_dec = reset_state(init_bwd_dec, batch_l)
	 local lossMLE = 0
	 --KLDloss
	 local kldloss = KLDloss:forward({mu_phi, logsigma_phi, mu_omega, logsigma_omega})/batch_l
	 --backward through kldloss
	 local dstats = KLDloss:backward({mu_phi, logsigma_phi, mu_omega, logsigma_omega})
	 dstats[1]:div(batch_l)
	 dstats[2]:div(batch_l)
	 dstats[3]:div(batch_l)
	 dstats[4]:div(batch_l)
        

	 for t = target_l, 1, -1 do
	    local pred = generator:forward(preds[t])
	    lossMLE = lossMLE + criterion:forward(pred, target_out[t])/batch_l
	    local dl_dpred = criterion:backward(pred, target_out[t])
	    dl_dpred:div(batch_l)
	    local dl_dtarget = generator:backward(preds[t], dl_dpred)
	    drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
	    local decoder_input
	    decoder_input = {target[t], z, table.unpack(rnn_state_dec[t-1])}
	    local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
	    -- accumulate encoder/decoder grads
	    dz:add(dlst[2])
	    drnn_state_dec[#drnn_state_dec]:zero()
	    if opt.input_feed == 1 then
	       drnn_state_dec[#drnn_state_dec]:add(dlst[3])
	    end	    
	    for j = dec_offset, #dlst do
	       drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
	    end	    
	 end
         word_vec_layers[3].gradWeight[1]:zero()
	 if opt.fix_word_vecs_dec == 1 then
	    word_vec_layers[3].gradWeight:zero()
	 end
	 
	 local grad_norm = 0
	 grad_norm = grad_norm + grad_params[3]:norm()^2 + grad_params[4]:norm()^2

         local dmu_phi, dlogsigma_phi = table.unpack(sampler:backward({mu_phi, logsigma_phi}, dz))
	 --add in dstats the ds based on the kldloss
	 dstats[1]:add(dmu_phi)
	 dstats[2]:add(dlogsigma_phi)

         -- backward prop through recognition_model
	 local dencoders = recognition_model:backward({context_source[{{},source_l}], context_target[{{},target_l}]}, {dstats[1],  dstats[2]})
	
	 -- backward prop thourgh prior model
	 local dx = prior_model:backward({context_source[{{},source_l}]}, {dstats[3], dstats[4]})
   

 	 grad_norm = grad_norm + grad_params[5]:norm()^2 + grad_params[6]:norm()^2

	 -- add gradients of inputs from prior and recognition
	 dencoders[1]:add(dx)

	 -- backward prop encoders
	 local drnn_state_enc_source = reset_state(init_bwd_enc, batch_l)
	 local drnn_state_enc_target = reset_state(init_bwd_enc, batch_l)

         encoder_grads_source[{{}, source_l}] = dencoders[1]:clone()
	 encoder_grads_target[{{}, target_l}] = dencoders[2]:clone()
         
	 -- backward prop encoder source
	 for t = source_l, 1, -1 do
	    local encoder_input = {source[t], table.unpack(rnn_state_enc_source[t-1])}
	    if t == source_l then
	       drnn_state_enc_source[#drnn_state_enc_source]:add(encoder_grads_source[{{},t}])
	    end
	    local dlst = encoder_clones_source[t]:backward(encoder_input, drnn_state_enc_source)
	    for j = 1, #drnn_state_enc_source do
	       drnn_state_enc_source[j]:copy(dlst[j+1])
	    end	    
	 end



	 -- backward prop encoder target
         for t = target_l, 1, -1 do
	    local encoder_input = {target[t], table.unpack(rnn_state_enc_target[t-1])}
	    if t == target_l then
	       drnn_state_enc_target[#drnn_state_enc_target]:add(encoder_grads_target[{{},t}])
	    end
	    local dlst = encoder_clones_target[t]:backward(encoder_input, drnn_state_enc_target)
	    for j = 1, #drnn_state_enc_target do
	       drnn_state_enc_target[j]:copy(dlst[j+1])
	    end
	 end
	 
	 	    	 
	 
         word_vec_layers[2].gradWeight[1]:zero()
	 if opt.fix_word_vecs_enc == 1 then
	    word_vec_layers[2].gradWeight:zero()
	 end
	 
	 word_vec_layers[1].gradWeight[1]:zero()
	 if opt.fix_word_vecs_enc == 1 then
	    word_vec_layers[1].gradWeight:zero()
	 end

	 grad_norm = grad_norm + grad_params[1]:norm()^2 + grad_params[2]:norm()^2
	 grad_norm = grad_norm^0.5	 
        
	 local new_grad_norm = 0
	 -- Shrink norm and update params
	 local param_norm = 0
	 local shrinkage = opt.max_grad_norm / grad_norm
	 for j = 1, #grad_params do
	    if shrinkage < 1 then
	       grad_params[j]:mul(shrinkage)
	    end
	    new_grad_norm = new_grad_norm + grad_params[j]:norm()^2
	   	    if opt.adagrad == 1 then
	       adagradStep(params[j], grad_params[j], layer_etas[j], optStates[j])
	    else
	       params[j]:add(grad_params[j]:mul(-opt.learning_rate))
	    end	    
	    param_norm = param_norm + params[j]:norm()^2
	 end	 
	 param_norm = param_norm^0.5
	 new_grad_norm = new_grad_norm^0.5

	 -- Bookkeeping
	 num_words_target = num_words_target + batch_l*target_l
	 num_words_source = num_words_source + batch_l*source_l
	 train_nonzeros = train_nonzeros + nonzeros
	 train_lossMLE = train_lossMLE + lossMLE*batch_l
	 train_kldloss = train_kldloss + kldloss
	 local time_taken = timer:time().real - start_time
         if i % opt.print_every == 0 then
	    local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
					epoch, i, data:size(), batch_l, opt.learning_rate)
	    stats = stats .. string.format('KLDloss: %.2f, PPL: %.2f, totalLoss: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
				  train_kldloss/i, math.exp(train_lossMLE/train_nonzeros), (train_kldloss/i)+ (train_lossMLE/train_nonzeros), param_norm, new_grad_norm)
	    stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
					   (num_words_target+num_words_source) / time_taken,
					   num_words_source / time_taken,
					   num_words_target / time_taken)			   
            print(stats)
         end
	 if i % 200 == 0 then
	    collectgarbage()
	 end
      end
      return train_lossMLE, train_nonzeros
   end   

   local total_loss, total_nonzeros, batch_loss, batch_nonzeros
   for epoch = opt.start_epoch, opt.epochs do
      generator:training()
      if opt.num_shards > 0 then
	 total_loss = 0
	 total_nonzeros = 0	 
	 local shard_order = torch.randperm(opt.num_shards)
	 for s = 1, opt.num_shards do
	    local fn = train_data .. '.' .. shard_order[s] .. '.hdf5'
	    print('loading shard #' .. shard_order[s])
	    local shard_data = data.new(opt, fn)
	    batch_loss, batch_nonzeros = train_batch(shard_data, epoch)
	    total_loss = total_loss + batch_loss
	    total_nonzeros = total_nonzeros + batch_nonzeros
	 end
      else
	 total_loss, total_nonzeros = train_batch(train_data, epoch)
	 print("Total loss after batch training ".. total_loss)
      end
      local train_score = math.exp(total_loss/total_nonzeros)
      print('Train', train_score)
      opt.train_perf[#opt.train_perf + 1] = train_score
      local score = eval(valid_data)
      opt.val_perf[#opt.val_perf + 1] = score
      if opt.adagrad == 0 then --unncessary with adagrad
	 decay_lr(epoch)
      end      
      -- clean and save models
      local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score)      
      if epoch % opt.save_every == 0 then
         print('saving checkpoint to ' .. savefile)
	 clean_layer(generator)
         torch.save(savefile, {{encoder_source, encoder_target, decoder, generator, recognition_model, prior_model, sampler, KLDloss}, opt})
      end      
   end
   -- save final model
   local savefile = string.format('%s_final.t7', opt.savefile)
   clean_layer(generator)
   print('saving final model to ' .. savefile)
   torch.save(savefile, {{encoder_source:double(), encoder_target:double(), decoder:double(), generator:double(), recognition_model:double(), prior_model:double(), sampler:double(), KLDloss:double()}, opt})
end

function eval(data)
   encoder_clones_source[1]:evaluate()   
   generator:evaluate()
   
   local nll = 0
   local total = 0
   for i = 1, data:size() do
      local d = data[i]
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l = d[5], d[6], d[7]
      local rnn_state_enc_source = reset_state(init_fwd_enc, batch_l)
      local context_source = context_proto[{{1, batch_l}, {1, source_l}}]
     
     -- forward prop encoder source
      for t = 1, source_l do
	 local encoder_input = {source[t], table.unpack(rnn_state_enc_source)}
	 local out = encoder_clones_source[1]:forward(encoder_input)
	 rnn_state_enc_source = out
	 context_source[{{},t}]:copy(out[#out])
      end	 
      
      -- use x to predict m and s from prior model
      local stats_omega = prior_model:forward(context_source[{{},source_l}])
      local mu_omega = stats_omega[1]
      local logsigma_omega = stats_omega[2]
      -- calculate z
      local z = sampler:forward({mu_omega, logsigma_omega})

      local rnn_state_dec = reset_state(init_fwd_dec, batch_l)


      
      local loss = 0
      for t = 1, target_l do
	 local decoder_input
	 decoder_input = {target[t], z, table.unpack(rnn_state_dec)}
	 local out = decoder_clones[1]:forward(decoder_input)
         rnn_state_dec = {}
	 if opt.input_feed == 1 then
	    table.insert(rnn_state_dec, out[#out])
	 end	 
         for j = 1, #out-1 do
	    table.insert(rnn_state_dec, out[j])
	 end
	 local pred = generator:forward(out[#out])
	 loss = loss + criterion:forward(pred, target_out[t])
      end
      nll = nll + loss
      total = total + nonzeros
   end
   local valid = math.exp(nll / total)
   print("Valid", valid)
   collectgarbage()
   return valid
end


function get_layer(layer)
   if layer.name ~= nil then
      if layer.name == 'word_vecs_dec' then
	 table.insert(word_vec_layers, layer)
      elseif layer.name == 'word_vecs_enc' then
	 table.insert(word_vec_layers, layer)
      elseif layer.name == 'charcnn_enc' or layer.name == 'mlp_enc' then
	 local p, gp = layer:parameters()
	 for i = 1, #p do
	    table.insert(charcnn_layers, p[i])
	    table.insert(charcnn_grad_layers, gp[i])
	 end	 
      end
   end
end

function main() 
    -- parse input params
   opt = cmd:parse(arg)
   if opt.gpuid >= 0 then
      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      if opt.gpuid2 >= 0 then
	 print('using CUDA on second GPU ' .. opt.gpuid2 .. '...')
      end      
      require 'cutorch'
      require 'cunn'
      if opt.cudnn == 1 then
	 print('loading cudnn...')
	 require 'cudnn'
      end      
      cutorch.setDevice(opt.gpuid)
      cutorch.manualSeed(opt.seed)      
   end
   
   -- Create the data loader class.
   print('loading data...')
   if opt.num_shards == 0 then
      train_data = data.new(opt, opt.data_file)
   else
      train_data = opt.data_file
   end
   
   valid_data = data.new(opt, opt.val_data_file)
   print('done!')
   print(string.format('Source vocab size: %d, Target vocab size: %d',
		       valid_data.source_size, valid_data.target_size))   
   opt.max_sent_l_src = valid_data.source:size(2)
   opt.max_sent_l_targ = valid_data.target:size(2)
   opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
   if opt.max_batch_l == '' then
      opt.max_batch_l = valid_data.batch_l:max()
   end
   
   print(string.format('Source max sent len: %d, Target max sent len: %d',
		       valid_data.source:size(2), valid_data.target:size(2)))   
   
   -- Build model
   if opt.train_from:len() == 0 then
      -- AL: ENCODER
      encoder_source = make_lstm(valid_data, opt, 'enc', 'source', opt.use_chars_enc)
      encoder_target = make_lstm(valid_data, opt, 'enc', 'target', opt.use_chars_enc)
      -- AL: DECODER
      decoder = make_lstm(valid_data, opt, 'dec', '', opt.use_chars_dec)
      -- AL: word generator
      generator, criterion = make_generator(valid_data, opt)
      -- AL: recognition model
      recognition_model = recognition_model1(opt.rnn_size, opt.rnn_size, opt.z_size) 
      -- AL: prior model
      prior_model = prior_model(opt.rnn_size, opt.z_size)
      -- AL: KDloss
      KLDloss = nn.KLDCriterion()
      KLDloss.sizeAverage = false
      -- AL: latent variable sampler
      sampler = nn.Sampler(opt.gpuid)
   else -- DON'T CARE
      assert(path.exists(opt.train_from), 'checkpoint path invalid')
      print('loading ' .. opt.train_from .. '...')
      local checkpoint = torch.load(opt.train_from)
      local model, model_opt = checkpoint[1], checkpoint[2]
      opt.num_layers = model_opt.num_layers
      opt.rnn_size = model_opt.rnn_size
      opt.input_feed = model_opt.input_feed
      opt._= model_opt.attn
      opt.brnn = model_opt.brnn
      encoder = model[1]:double()
      decoder = model[2]:double()      
      generator = model[3]:double()
      _, criterion = make_generator(valid_data, opt)
   end   
   
   layers = {encoder_source, encoder_target, decoder, generator, recognition_model, prior_model}

   -- AL: Adagrad stuff
   if opt.adagrad == 1 then
      layer_etas = {}
      optStates = {}
      for i = 1, #layers do
	 layer_etas[i] = opt.learning_rate
	 optStates[i] = {}
      end     
   end
   
   -- AL: Initialize gpu stuff
   if opt.gpuid >= 0 then
      for i = 1, #layers do	 
	 layers[i]:cuda()
      end
      KLDloss:cuda()
      sampler:cuda()
      criterion:cuda()      
   end

   -- these layers will be manipulated during training
   word_vec_layers = {}
   -- AL: WHAT??
   encoder_source:apply(get_layer)   
   encoder_target:apply(get_layer)
   decoder:apply(get_layer)
   train(train_data, valid_data)
end

main()

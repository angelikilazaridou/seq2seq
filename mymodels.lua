function nn.Module:reuseMem()
   self.reuse = true
   return self
end

function nn.Module:setReuse()
   if self.reuse then
      self.gradInput = self.output
   end
end

function compute_z()
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- mu
    table.insert(inputs, nn.Identity()()) -- sigma
    table.insert(inputs, nn.Identity()()) -- epsilon

    local outputs = {}
    table.insert(outputs, nn.CAddTable()({inputs[1], nn.CMulTable()({inputs[2],inputs[3]})}))

    return nn.gModule(inputs, outputs)
end

function recognition_model1(x_size, y_size, z_size)
 
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    table.insert(inputs, nn.Identity()()) -- y


    local x = nn.JoinTable(2)({inputs[1], inputs[2]})
    
    -- get mean and variance
    local m = nn.Linear(x_size + y_size, z_size)(x)
    local s = nn.Linear(x_size + y_size, z_size)(x)

    local outputs = {}
    table.insert(outputs, m)
    table.insert(outputs, s)

    return nn.gModule(inputs, outputs)

end

function recognition_model3(x_size, y_size, h_size, z_size)
 
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    table.insert(inputs, nn.Identity()()) -- y


    local x = nn.JoinTable(2)({inputs[1], inputs[2]})
    local h = nn.ReLU()(nn.Linear(x_size+y_size, h_size)(x))

    -- get mean and variance
    local m = nn.Linear(h_size, z_size)(h)
    local s = nn.Linear(h_size, z_size)(h)

    local outputs = {}
    table.insert(outputs, m)
    table.insert(outputs, s)

    return nn.gModule(inputs, outputs)

end


function recognition_model2(x_size, z_size)
 
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x
    table.insert(inputs, nn.Identity()()) -- y


    local x = nn.CSubTable()({inputs[1], inputs[2]})
    
    -- get mean and variance
    local m = nn.Linear(x_size, z_size)(x)
    local s = nn.Linear(x_size, z_size)(x)

    local outputs = {}
    table.insert(outputs, m)
    table.insert(outputs, s)

    return nn.gModule(inputs, outputs)

end


function prior_model(x_size, z_size)
    
    local inputs = {}
    table.insert(inputs, nn.Identity()()) -- x

    local x = inputs[1]

    -- get mean and variance
    local m = nn.Linear(x_size, z_size)(x)
    local s = nn.Linear(x_size, z_size)(x)
    
    local outputs = {}
    table.insert(outputs, m)
    table.insert(outputs, s)
    
    return nn.gModule(inputs, outputs)

end

function make_lstm(data, opt, model, side, use_chars)
   assert(model == 'enc' or model == 'dec')
   local name = '_' .. model
   local dropout = opt.dropout or 0
   local n = opt.num_layers
   local rnn_size = opt.rnn_size
   local z_size = opt.z_size
   local input_size
   if use_chars == 0 then
      input_size = opt.word_vec_size
   else
      input_size = opt.num_kernels
   end   
   local offset = 0
  -- there will be 2*n+3 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
   if model == 'dec' then
      table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
      offset = offset + 1
      if opt.input_feed == 1 then
	 table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
	 offset = offset + 1
      end
   end
   for L = 1,n do
      table.insert(inputs, nn.Identity()()) -- prev_c[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end

   local x, input_size_L
   local outputs = {}
  for L = 1,n do
     -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]    
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
       if use_chars == 0 then
	  local word_vecs
	  if model == 'enc' and side == 'source' then
	     word_vecs = nn.LookupTable(data.source_size, input_size)
	  else
	     word_vecs = nn.LookupTable(data.target_size, input_size)
	  end	  
	  word_vecs.name = 'word_vecs' .. name
	  x = word_vecs(inputs[1]) -- batch_size x word_vec_size
       end
       input_size_L = input_size
       if model == 'dec' then
	  if opt.input_feed == 1 then
	     x = nn.JoinTable(2)({x, inputs[1+offset]}) -- batch_size x (word_vec_size + rnn_size)
	     input_size_L = input_size + rnn_size
	  end	  
       end
    else
       x = outputs[(L-1)*2]
       if opt.res_net == 1 and L > 2 then
	  x = nn.CAddTable()({x, outputs[(L-2)*2]})       
       end       
       input_size_L = rnn_size
       if opt.multi_attn == L and model == 'dec' then
	  local multi_attn = make_decoder_attn(data, opt, 1)
	  multi_attn.name = 'multi_attn' .. L
	  x = multi_attn({x, inputs[2]})
       end
       if dropout > 0 then
	  x = nn.Dropout(dropout, nil, false)(x)
       end       
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if model == 'dec' then
     local top_h = outputs[#outputs]
     local decoder_out
     decoder_out = nn.JoinTable(2)({top_h, inputs[2]})
     decoder_out = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2, opt.rnn_size)(decoder_out))
     if dropout > 0 then
	decoder_out = nn.Dropout(dropout, nil, false)(decoder_out)
     end     
     table.insert(outputs, decoder_out)
  end
  return nn.gModule(inputs, outputs)
end


function make_generator(data, opt)
   local model = nn.Sequential()
   model:add(nn.Linear(opt.rnn_size, data.target_size))
   model:add(nn.LogSoftMax())
   local w = torch.ones(data.target_size)
   w[1] = 0
   criterion = nn.ClassNLLCriterion(w)
   criterion.sizeAverage = false
   return model, criterion
end



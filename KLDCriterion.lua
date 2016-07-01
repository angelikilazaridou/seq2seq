local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(mean_phi, log_var_phi, mean_omega, log_var_omega)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    local mean_phi_sq = torch.pow(mean_phi, 2)
    local mean_omega_sq = torch.pow(mean_omega, 2)

    --term A
    local KLDelements = log_var_omega:clone()
    KLDelements:add(-1, log_var_phi)
    
    --term B
    KLDelements:add(-1)
   
    --term C
    local termC = torch.exp(log_var_phi)
    termC:add(mean_phi_sq)
    termC:add(mean_omega_sq)
    
    local tmp = mean_omega
    tmp:mul(mean_phi):mul(2)

    termC:add(tmp)
    termC:div(torch.exp(log_var_omega))

    KLDelements:add(-1, termC)
    
    self.output = - 0.5 * torch.sum(KLDelements)
    
    return self.output
end



function KLDCriterion:updateGradInput(mean_phi, log_var_phi, mean_omega, log_var_omega)

    self.gradInput = {}
    
    self.gradInput[1] = mean_phi:clone() + mean_omega:clone()
    self.gradInput[1]:div(torch.exp(log_var_omega))
    
    self.gradInput[3] = mean_omega:clone() + mean_phi:clone()
    self.gradInput[3]:div(torch.exp(log_var_omega))

    self.gradInput[2] = 1
    self.gradInput[2]:div(torc.exp(log_var_phi)):mul(-0.5)
    local tmp = 1
    tmp:div(log_var_omega)
    self.gradInput[2]:add(tmp)

    self.gradInput[4] = -0.5
    self.gradInput[4]:div(torch.exp(log_var_omega))
 
    local tmp = torch.exp(log_var_phi)
    tmp:add(mean_phi_sq):add(mean_omega_sq)
    local tmp2 = mean_phi:clone()
    tmp2:mul(2):mul(mean_omega)
    tmp:add(tmp2):div(torch.pow(torch.exp(log_var_omega),2)):mul(-1):mul(torch.exp(log_var_omega)):mul(0.5)

    self.gradInput[4]:add(tmp)


    return self.gradInput
end



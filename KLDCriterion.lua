local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(inputs)

    mu_phi, logsigma_phi, mu_omega, logsigma_omega = table.unpack(inputs)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
   
    print(mu_phi:size())
    print(logsigma_phi:size())
    print(logsigma_omega:size())
    local mu_phi_sq = torch.pow(mu_phi, 2)
    local mu_omega_sq = torch.pow(mu_omega, 2)

    --term A
    local KLDelements = logsigma_omega:clone()
    KLDelements:add(-1, logsigma_phi)
    
    --term B
    KLDelements:add(-1)
   
    --term C
    local termC = torch.exp(logsigma_phi)
    termC:add(mu_phi_sq)
    termC:add(mu_omega_sq)
    
    local tmp = mu_omega
    tmp:cmul(mu_phi):mul(2)

    termC:add(tmp)
    termC:cdiv(torch.exp(logsigma_omega))

    KLDelements:add(-1, termC)
    
    self.output = - 0.5 * torch.sum(KLDelements)
    
    return self.output
end



function KLDCriterion:updateGradInput(inputs)
    
    mu_phi, logsigma_phi, mu_omega, logsigma_omega = table.unpack(inputs)

    self.gradInput = {}
    
    self.gradInput[1] = mu_phi:clone() + mu_omega:clone()
    self.gradInput[1]:cdiv(torch.exp(logsigma_omega))
    
    self.gradInput[3] = mu_omega:clone() + mu_phi:clone()
    self.gradInput[3]:cdiv(torch.exp(logsigma_omega))

    local allOnes = logsigma_phi:clone():fill(1)

    self.gradInput[2] = allOnes:clone()
    self.gradInput[2]:cdiv(torch.exp(logsigma_phi)):mul(-0.5)

    local tmp = allOnes:clone()
    tmp:cdiv(logsigma_omega)
    self.gradInput[2]:add(tmp)

  
    self.gradInput[4] = allOnes:clone():fill(-0.5)
    self.gradInput[4]:cdiv(torch.exp(logsigma_omega))
 
    local tmp = torch.exp(logsigma_phi)
    tmp:add(torch.pow(mu_phi,2)):add(torch.pow(mu_omega,2))
    local tmp2 = mu_phi:clone()
    tmp2:mul(2):cmul(mu_omega)
    tmp:add(tmp2):cdiv(torch.pow(torch.exp(logsigma_omega),2)):mul(-1):cmul(torch.exp(logsigma_omega)):mul(0.5)

    self.gradInput[4]:add(tmp)


    return self.gradInput
end



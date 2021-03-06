local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(inputs)

    
    mu_phi, logsigma_phi, mu_omega, logsigma_omega = table.unpack(inputs)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
   

    --term A
    local termA = mu_phi:clone()
    termA:add(-1, mu_omega)
    termA = torch.pow(termA, 2)
    termA:cdiv(torch.exp(logsigma_omega))
    self.termA = termA:clone()


    local KLDelements = termA:clone()

    local termB = torch.exp(logsigma_phi)
    termB:cdiv(torch.exp(logsigma_omega)):add(-1):add(-1, logsigma_phi):add(logsigma_omega)
    
    --term B
    KLDelements:add(termB)

    -- sum over dimensions
    self.output =  0.5 * torch.sum(KLDelements)
    return self.output
end



function KLDCriterion:updateGradInput(inputs)
    mu_phi, logsigma_phi, mu_omega, logsigma_omega = table.unpack(inputs)

    self.gradInput = {}
    
    self.gradInput[1] = mu_phi:clone() - mu_omega:clone()
    self.gradInput[1]:cdiv(torch.exp(logsigma_omega))
     
    self.gradInput[3] = mu_omega:clone() -  mu_phi:clone()
    self.gradInput[3]:cdiv(torch.exp(logsigma_omega))

    local tmp = logsigma_phi:clone()
    tmp:add(-1, logsigma_omega)
    tmp = torch.exp(tmp)
    tmp:add(-1):mul(0.5)

    self.gradInput[2] = tmp:clone()

   
    local tmp = logsigma_phi:clone()
    tmp:add(-1, logsigma_omega)
    tmp = torch.exp(tmp)
    tmp:mul(-1):add(1)
    tmp:add(-1, self.termA):mul(0.5)
    
    self.gradInput[4] = tmp:clone()

 
    return self.gradInput
end



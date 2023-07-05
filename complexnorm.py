import torch
import torch.nn as nn

class ComplexBatchNorm1d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,3))
            self.bias = nn.Parameter(torch.Tensor(num_features).type(torch.complex64))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2],1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)

    def forward(self, input_x):
        assert(len(input_x.shape) == 3)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input_x.mean([0, 2])

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            input_x = input_x - mean[None, :, None]

            # Elements of the covariance matrix (biased for train)
            n = input_x.numel() / input_x.size(1)
            Crr = 1./n*input_x.real.pow(2).sum(dim=[0,2])+self.eps
            Cii = 1./n*input_x.imag.pow(2).sum(dim=[0,2])+self.eps
            Cri = (input_x.real.mul(input_x.imag)).mean(dim=[0,2])

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps

            input_x = input_x-mean[None,:,None]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None,:,None]*input_x.real+Rri[None,:,None]*input_x.imag, \
                           Rii[None,:,None]*input_x.imag+Rri[None,:,None]*input_x.real

        if self.affine:
            input_r, input_i = self.weight[None,:,0,None]*input_r+self.weight[None,:,2,None]*input_i,\
                               self.weight[None,:,2,None]*input_r+self.weight[None,:,1,None]*input_i
            output = torch.complex(input_r, input_i) + self.bias[None,:,None]
        else:
            output = torch.complex(input_r, input_i)

        return output

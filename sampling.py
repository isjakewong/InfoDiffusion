import torch 

class DiffusionProcess():
    def __init__(self, args, diffusion_fn, device, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape
        self.deterministic = args.deterministic
        self.a_dim = args.a_dim
        self.model = args.model
        
        self.diffusion_fn = diffusion_fn.to(device=device)
        self.device = device

    
    def _ddpm_one_diffusion_step(self, x, a=None):
        '''
        x   : perturbated data
        '''
        for idx in reversed(range(len(self.alpha_bars))):

            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            if self.model == 'vanilla':
                predict_epsilon = self.diffusion_fn(x, idx)
            else:
                predict_epsilon = self.diffusion_fn(x, idx, a)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)

            x = mu_theta_xt + sqrt_tilde_beta * noise

            yield x
    
    def _ddim_one_diffusion_step(self, x, a=None):
        '''
        x   : perturbated data
        '''
        eta = 0.01
        for idx in reversed(range(len(self.alpha_bars))):

            if self.model == 'vanilla':
                predict_epsilon = self.diffusion_fn(x, idx)
            else:
                predict_epsilon = self.diffusion_fn(x, idx, a)
            x_0 = (x - torch.sqrt(1 - self.alpha_prev_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_prev_bars[idx])
            if idx == 0:
                x = x_0
            else:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - self.alpha_prev_bars[idx-1]) / (1 - self.alpha_bars[idx-1])) * torch.sqrt(self.betas[idx-1])
                x = torch.sqrt(self.alpha_prev_bars[idx-1]) * x_0 + torch.sqrt(1 - self.alpha_prev_bars[idx-1] - sigma**2) * predict_epsilon
                x += sigma * noise
            yield x

    def _ddim_one_reverse_diffusion_step(self, x, a=None):
        for idx in range(len(self.alpha_bars)-1):
            if idx == 0:
                yield x
            else:
                if self.model == 'vanilla':
                    predict_epsilon = self.diffusion_fn(x, idx)
                else:
                    predict_epsilon = self.diffusion_fn(x, idx, a)
                x_0 = (x - torch.sqrt(1 - self.alpha_prev_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_prev_bars[idx])
                x = torch.sqrt(self.alpha_prev_bars[idx+1]) * x_0 + torch.sqrt(1 - self.alpha_prev_bars[idx+1]) * predict_epsilon
                yield x

    def _one_diffusion_step(self, sample, a=None, deterministic=False):
        if deterministic:
            return self._ddim_one_diffusion_step(sample, a)
        else:
            return self._ddpm_one_diffusion_step(sample, a)

    @torch.no_grad()
    def reverse_sampling(self, x0, a=None):
        sample = x0
        for sample in self._ddim_one_reverse_diffusion_step(sample):
            final = sample

        return final

    @torch.no_grad()
    def sampling(self, sampling_number=16, xT=None, a=None):
        if xT == None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        if self.model != 'vanilla':
            if a == None:
                a = torch.randn([sampling_number, self.a_dim]).to(device=self.device)

        sample = xT
        for sample in self._one_diffusion_step(sample=sample, a=a, deterministic=self.deterministic):
            final = sample

        return final


class TwoPhaseDiffusionProcess():
    def __init__(self, args, diffusion_fn_1, diffusion_fn_2, device, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape
        self.deterministic = args.deterministic
        self.a_dim = args.a_dim
        self.model = args.model
        self.split_step = args.split_step
        self.mode = args.mode
        
        self.diffusion_fn_1 = diffusion_fn_1.to(device=device)
        self.diffusion_fn_2 = diffusion_fn_2.to(device=device)
        self.device = device

    
    def _ddpm_one_diffusion_step(self, x, a=None, t=None):
        '''
        x   : perturbated data
        '''
        for idx in reversed(range(len(self.alpha_bars))):

            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            if t <= self.split_step:
                predict_epsilon = self.diffusion_fn_2(x, idx)
            else:
                predict_epsilon = self.diffusion_fn_1(x, idx, a)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)

            x = mu_theta_xt + sqrt_tilde_beta * noise

            yield x
    
    def _ddim_one_diffusion_step(self, x, a=None, t=None):
        '''
        x   : perturbated data
        '''
        eta = 0.01
        for idx in reversed(range(len(self.alpha_bars))):

            if t <= self.split_step:
                predict_epsilon = self.diffusion_fn_2(x, idx)
            else:
                predict_epsilon = self.diffusion_fn_1(x, idx, a)
            x_0 = (x - torch.sqrt(1 - self.alpha_prev_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_prev_bars[idx])
            if idx == 0:
                x = x_0
            else:
                noise = torch.randn_like(x)
                sigma = eta * torch.sqrt((1 - self.alpha_prev_bars[idx-1]) / (1 - self.alpha_bars[idx-1])) * torch.sqrt(self.betas[idx-1])
                x = torch.sqrt(self.alpha_prev_bars[idx-1]) * x_0 + torch.sqrt(1 - self.alpha_prev_bars[idx-1] - sigma**2) * predict_epsilon
                x += sigma * noise
            yield x

    def _ddim_one_reverse_diffusion_step(self, x, a=None):
        for idx in range(len(self.alpha_bars)-1):
            if idx == 0:
                yield x
            else:
                if t <= self.split_step:
                    predict_epsilon = self.diffusion_fn_2(x, idx)
                else:
                    predict_epsilon = self.diffusion_fn_1(x, idx, a)
                x_0 = (x - torch.sqrt(1 - self.alpha_prev_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_prev_bars[idx])
                x = torch.sqrt(self.alpha_prev_bars[idx+1]) * x_0 + torch.sqrt(1 - self.alpha_prev_bars[idx+1]) * predict_epsilon
                yield x

    def _one_diffusion_step(self, sample, a=None, deterministic=False, t=None):
        if deterministic:
            return self._ddim_one_diffusion_step(sample, a, t)
        else:
            return self._ddpm_one_diffusion_step(sample, a, t)

    @torch.no_grad()
    def reverse_sampling(self, x0, a=None):
        sample = x0
        for sample in self._ddim_one_reverse_diffusion_step(sample):
            final = sample

        return final

    @torch.no_grad()
    def sampling(self, sampling_number=16, xT=None, a=None):
        if xT == None:
            xT = torch.randn([sampling_number, *self.shape]).to(device=self.device)
        if a == None:
            a = torch.randn([sampling_number, self.a_dim]).to(device=self.device)

        sample = xT
        t = 0
        for sample in self._one_diffusion_step(sample=sample, a=a, deterministic=self.deterministic, t=t):
            final = sample
            t += 1

        return final
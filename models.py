import os
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from modules import *
from utils import compute_kernel, compute_mmd

class UNet(nn.Module):
    def __init__(self, T, ch=64, ch_mult=[1,2,4,8], attn=[2], num_res_blocks=2, dropout=0.1, a_dim=10, shape=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.a_dim = a_dim
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(shape[0], ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True, crossattn=False),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False, crossattn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, shape[0], 3, stride=1, padding=1)
        )

        self.fc_a = nn.Linear(self.a_dim, tdim)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.fc_a.weight)
        init.zeros_(self.fc_a.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            if isinstance(layer, ResBlock):
                h = layer(h, temb)
            else:
                h = layer(h)
        
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

class AuxiliaryUNet(nn.Module):
    def __init__(self, T, ch=64, ch_mult=[1,2,4,8], attn=[2], num_res_blocks=2, dropout=0.1, a_dim=10, shape=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.a_dim = a_dim
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(shape[0], ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(AuxResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            AuxResBlock(now_ch, now_ch, tdim, dropout, attn=True, crossattn=False),
            AuxResBlock(now_ch, now_ch, tdim, dropout, attn=False, crossattn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(AuxResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, shape[0], 3, stride=1, padding=1)
        )

        self.fc_a = nn.Linear(self.a_dim, tdim)

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.fc_a.weight)
        init.zeros_(self.fc_a.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, a):
        # Latent embedding
        aemb = self.fc_a(a)

        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, temb, aemb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            if isinstance(layer, AuxResBlock):
                h = layer(h, temb, aemb)
            else:
                h = layer(h)
        
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, AuxResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, aemb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

class Encoder(nn.Module):
    def __init__(self, ch=64, ch_mult=[1,2,4,8,8], attn=[2], num_res_blocks=2, dropout=0.1, a_dim=1, shape=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        self.shape = shape
        self.a_dim = a_dim
        self.head = nn.Conv2d(shape[0], ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock_encoder(
                    in_ch=now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock_encoder(now_ch, now_ch, dropout, attn=True),
            ResBlock_encoder(now_ch, now_ch, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock_encoder(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, 1, 3, stride=1, padding=1)
        )

        self.fc_a = nn.Linear(self.shape[1]*self.shape[2], self.a_dim)
        self.fc_mu = nn.Linear(self.a_dim, self.a_dim)
        self.fc_var = nn.Linear(self.a_dim, self.a_dim)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.fc_a.weight)
        init.zeros_(self.fc_a.bias)
        init.xavier_uniform_(self.fc_mu.weight)
        init.zeros_(self.fc_mu.bias)
        init.xavier_uniform_(self.fc_var.weight)
        init.zeros_(self.fc_var.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x):
        # Downsampling
        h = self.head(x)

        hs = [h]
        for layer in self.downblocks:
            if isinstance(layer, ResBlock_encoder):
                h = layer(h)
            else:
                h = layer(h, 0, 0) # for downsample module, 0 is placeholder for temb and aemb
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock_encoder):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h)
            else:
                h = layer(h, 0, 0) # for upsample module, 0 is placeholder for temb and aemb
        
        h = torch.flatten(self.tail(h), start_dim=1)

        a = self.fc_a(h)
        mu = self.fc_mu(a)
        log_var = self.fc_var(a)
        a_q = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

        assert len(hs) == 0
        return a, a_q, mu, log_var


class Decoder(nn.Module):
    def __init__(self, ch=64, ch_mult=[1,2,4,8], attn=[2], num_res_blocks=2, dropout=0.1, a_dim=10, shape=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        self.a_dim = a_dim
        self.shape = shape
        self.head = nn.Conv2d(shape[0], ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock_encoder(
                    in_ch=now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock_encoder(now_ch, now_ch, dropout, attn=True),
            ResBlock_encoder(now_ch, now_ch, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock_encoder(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, shape[0], 3, stride=1, padding=1)
        )

        self.fc_a = nn.Linear(self.a_dim, self.shape[0]*self.shape[1]*self.shape[2])

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, a):
        # Latent embedding
        aemb = self.fc_a(a)
        h = aemb.reshape(a.shape[0], self.shape[0], self.shape[1], self.shape[2])

        # Downsampling
        h = self.head(h)
        hs = [h]
        for layer in self.downblocks:
            if isinstance(layer, ResBlock_encoder):
                h = layer(h)
            else:
                h = layer(h, 0) # for downsample module, 0 is placeholder
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock_encoder):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h)
            else:
                h = layer(h, 0) # for upsample module, 0 is placeholder
        rec = self.tail(h)

        assert len(hs) == 0
        return rec

class InfoDiff(nn.Module):
    def __init__(self, args, device, shape):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.device = device
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps), dim=0).to(device=device)
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps).to(device = device)
        self.alphas = 1 - self.betas
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        if args.input_size == 28:
            ch_mult = [1,2,4,]
        else:
            ch_mult = [1,2,4,8]
        self.backbone = AuxiliaryUNet(ch_mult=ch_mult, T=args.diffusion_steps, ch=args.unets_channels, a_dim=args.a_dim, shape=shape)
        self.encoder = Encoder(ch_mult=ch_mult, ch=args.encoder_channels, a_dim=args.a_dim, shape=shape)
        self.mmd_weight : float = args.mmd_weight
        self.kld_weight : float = args.kld_weight
        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''
        output, epsilon, a, mu, log_var = self.forward(x, idx=idx, get_target=True)
        # denoising matching term
        loss = (output - epsilon).square().mean()
        print('denoising loss:', loss)
        # reconstruction term
        x_0 = torch.sqrt(1 / self.alphas[0]) * (x - self.betas[0] / torch.sqrt(1 - self.alpha_bars[0]) * output)
        loss_rec = (x_0 - x).square().mean()
        loss += loss_rec / args.diffusion_steps
        print('recon loss:', loss_rec / args.diffusion_steps)
        if args.mmd_weight != 0:
            # MMD term
            true_samples = torch.randn_like(a, device=self.device)
            loss_mmd = compute_mmd(true_samples, a)
            print('mmd loss:', args.mmd_weight * loss_mmd)
            loss += args.mmd_weight * loss_mmd
        elif args.kld_weight != 0:
            # KLD term
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            if args.use_C:
                # KLD term w/ control constant
                self.C_max = torch.FloatTensor([args.C_max]).to(device=self.device)
                C = torch.clamp(self.C_max/args.epochs*curr_epoch, torch.FloatTensor([0]).to(device=self.device), self.C_max)
                loss += args.kld_weight * (kld_loss - C.squeeze(dim=0)).abs()
            else:
                print('kld loss:', args.kld_weight * kld_loss)
                loss += args.kld_weight * kld_loss

        return loss
        
    def forward(self, x, idx=None, a=None, get_target=False):

        if idx is None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0), )).to(device = self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None, None, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device = self.device).long()
            x_tilde = x

        if not (self.mmd_weight != 0 and self.kld_weight != 0):
            if a is None:
                a, a_q, mu, log_var = self.encoder(x)
            else:
                a_q = a
        
        if self.mmd_weight != 0 and self.kld_weight != 0:
            assert bool(self.mmd_weight == 0) ^ bool(self.kld_weight == 0), f"Error: not implemented for both mmd_weight AND kld_weight: {self.mmd_weight, self.kld_weight}"
        elif self.mmd_weight == 0 and self.kld_weight == 0:
            output = self.backbone(x_tilde, idx, a)
        elif self.mmd_weight != 0:
            output = self.backbone(x_tilde, idx, a)
        elif self.kld_weight != 0:
            output = self.backbone(x_tilde, idx, a_q)

        return (output, epsilon, a, mu, log_var) if get_target else output


class Diff(nn.Module):
    def __init__(self, args, device, shape):
        '''
        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        '''

        super().__init__()
        self.device = device
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps), dim=0).to(device=device)
        self.betas = torch.linspace(start=args.beta1, end=args.betaT, steps=args.diffusion_steps).to(device = device)
        self.alphas = 1 - self.betas
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        if args.input_size == 28:
            ch_mult = [1,2,4,]
        else:
            ch_mult = [1,2,4,8]
        self.backbone = UNet(ch_mult=ch_mult, T=args.diffusion_steps, ch=args.unets_channels, a_dim=args.a_dim, shape=shape)
        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''
        output, epsilon = self.forward(x, idx=idx, get_target=True)
        # denoising matching term
        loss = (output - epsilon).square().mean()
        return loss
        
    def forward(self, x, idx=None, a=None, get_target=False):

        if idx is None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0), )).to(device = self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None, None, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon
        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device = self.device).long()
            x_tilde = x

        output = self.backbone(x_tilde, idx)

        return (output, epsilon) if get_target else output

class VAE(nn.Module):
    def __init__(self, args, device, shape):
        super().__init__()
        self.device = device
        if args.input_size == 28:
            ch_mult = [1,2,4,]
        else:
            ch_mult = [1,2,4,8]
        self.encoder = Encoder(ch_mult=ch_mult, ch=args.encoder_channels, a_dim=args.a_dim, shape=shape)
        self.decoder = Decoder(ch_mult=ch_mult, ch=args.encoder_channels, a_dim=args.a_dim, shape=shape)
        self.mmd_weight : float = args.mmd_weight
        self.kld_weight : float = args.kld_weight
        self.to(device)

    def loss_fn(self, args, x, curr_epoch=0):
        reconstruction, a, mu, log_var = self.forward(x, get_target=True)
        # denoising matching term
        loss = (reconstruction - x).square().mean()
        print('reconstruction loss:', loss)
        # prior matching term
        if args.mmd_weight != 0:
            # MMD term
            true_samples = torch.randn_like(a, device=self.device)
            loss_mmd = compute_mmd(true_samples, a)
            print('mmd loss:', args.mmd_weight * loss_mmd)
            loss += args.mmd_weight * loss_mmd
        elif args.kld_weight != 0:
            # KLD term
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            if args.use_C:
                # KLD term w/ control constant
                self.C_max = torch.FloatTensor([args.C_max]).to(device=self.device)
                C = torch.clamp(self.C_max/args.epochs*curr_epoch, torch.FloatTensor([0]).to(device=self.device), self.C_max)
                loss += args.kld_weight * (kld_loss - C.squeeze(dim=0)).abs()
                print('kld-c loss:', (kld_loss - C.squeeze(dim=0)).abs())
            else:
                loss += args.kld_weight * kld_loss
                print('kld loss:', args.kld_weight * kld_loss)
            

        return loss
        
    def forward(self, x, get_target=False):
        a, a_q, mu, log_var = self.encoder(x)
        
        if self.mmd_weight != 0 and self.kld_weight != 0:
            assert bool(self.mmd_weight == 0) ^ bool(self.kld_weight == 0), f"Error: not implemented for both mmd_weight AND kld_weight: {self.mmd_weight, self.kld_weight}"
        elif self.mmd_weight == 0 and self.kld_weight == 0:
            reconstruction = self.decoder(a)
        elif self.mmd_weight != 0:
            reconstruction = self.decoder(a)
        elif self.kld_weight != 0:
            reconstruction = self.decoder(a_q)

        return (reconstruction, a, mu, log_var) if get_target else reconstruction


class FeatureClassfier(nn.Module):
    def __init__(self, args, output_dim = 40):
        super(FeatureClassfier, self).__init__()

        """build full connect layers for every attribute"""
        self.fc_set = {}

        self.fc = nn.Sequential(
            nn.Linear(args.a_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        res = self.fc(x)
        res = self.sigmoid(res)
        return res
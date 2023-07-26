import torch
import torch.nn as nn
from torch.nn import init
from modules import *
from utils import compute_mmd, gaussian_mixture, swiss_roll

class UNet(nn.Module):
    def __init__(self, T, ch=64, ch_mult=[1,2,4,8], attn=[2], num_res_blocks=2, dropout=0.1, shape=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
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

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
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


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        activation: str = None,
        cond_channels: int = None,
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        if self.activation is not None:
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == 'leaky_relu':
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == 'silu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)

            # scale shift first
            x = x * (self.condition_bias + cond)

            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class LatentUNet(nn.Module):
    def __init__(self, T, num_layers=10, dropout=0.1, shape=None, activation='silu', 
                 num_time_emb_channels: int = 64, num_time_layers: int = 2):
        super().__init__()
        self.num_time_emb_channels = num_time_emb_channels
        self.shape = shape

        layers = []
        for i in range(num_time_layers):
            if i == 0:
                a = num_time_emb_channels
                b = shape[-1]
            else:
                a = shape[-1]
                b = shape[-1]
            layers.append(nn.Linear(a, b))
            if i < num_time_layers - 1:
                layers.append(nn.SiLU())
        self.time_embed = nn.Sequential(*layers)

        self.skip_layers = list(range(1, num_layers))
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = activation
                norm = True
                cond = True
                a, b = shape[-1], shape[-1] * 4
                dropout = dropout
            elif i == num_layers - 1:
                act = None
                norm = False
                cond = False
                a, b = shape[-1] * 4, shape[-1]
                dropout = 0
            else:
                act = 'silu'
                norm = True
                cond = True
                a, b = shape[-1] * 4, shape[-1] * 4
                dropout = dropout

            if i in self.skip_layers:
                a += shape[-1]

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=shape[-1],
                    use_cond=cond,
                    condition_bias=1,
                    dropout=dropout,
                ))

    def forward(self, x, t):
        # Timestep embedding
        t = timestep_embedding(t, self.num_time_emb_channels)
        temb = self.time_embed(t)

        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=temb)
        return h


class AuxiliaryUNet(nn.Module):
    def __init__(self, T, ch=64, ch_mult=[1,2,4,8], attn=[2], num_res_blocks=2, dropout=0.1, a_dim=32, shape=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.a_dim = a_dim
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.fc_a = nn.Linear(self.a_dim, tdim)

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


class BottleneckAuxUNet(nn.Module):
    def __init__(self, T, ch=64, ch_mult=[1,2,4,8], attn=[2], num_res_blocks=2, dropout=0.1, a_dim=32, shape=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.a_dim = a_dim
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.fc_a = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.a_dim, tdim)
        )
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
            AuxResBlock(now_ch, now_ch, tdim, dropout, attn=True, crossattn=False),
            AuxResBlock(now_ch, now_ch, tdim, dropout, attn=False, crossattn=False),
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

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.kaiming_normal_(self.fc_a[1].weight,
                             a=0,
                             nonlinearity='relu')
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
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            if isinstance(layer, AuxResBlock):
                h = layer(h, temb, aemb)
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


class Encoder(nn.Module):
    def __init__(self, ch=64, ch_mult=[1,2,4,8,8], attn=[2], num_res_blocks=2, dropout=0.1, a_dim=32, shape=None):
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
                h = layer(h, None, None) # for downsample module, 0 is placeholder for temb and aemb
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
                h = layer(h, None, None) # for upsample module, 0 is placeholder for temb and aemb

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
                h = layer(h, None) # for downsample module, 0 is placeholder
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
                h = layer(h, None) # for upsample module, 0 is placeholder
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
            ch_mult = [1,2,2,2]
        if args.is_bottleneck:
            self.backbone = BottleneckAuxUNet(ch_mult=ch_mult, T=args.diffusion_steps, ch=args.unets_channels, a_dim=args.a_dim, shape=shape)
        else:
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

        if self.mmd_weight != 0 and self.kld_weight != 0:
            # MMD term
            if args.prior == 'regular':
                true_samples = torch.randn_like(a, device=self.device)
            elif args.prior == '10mix':
                prior = gaussian_mixture(args.batch_size, args.a_dim)
                true_samples = torch.FloatTensor(prior).to(device=self.device)
            elif args.prior == 'roll':
                prior = swiss_roll(args.batch_size)
                true_samples = torch.FloatTensor(prior).to(device=self.device)
            loss_mmd = compute_mmd(true_samples, mu)
            print('mmd loss:', args.mmd_weight * loss_mmd)
            loss += args.mmd_weight * loss_mmd
            # KLD term
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            if args.use_C:
                # KLD term w/ control constant
                self.C_max = torch.FloatTensor([args.C_max]).to(device=self.device)
                C = torch.clamp(self.C_max/args.epochs*curr_epoch, torch.FloatTensor([0]).to(device=self.device), self.C_max)
                loss += args.kld_weight * (kld_loss - C.squeeze(dim=0)).abs()
            else:
                print('kld loss:', args.kld_weight * kld_loss)
                loss += args.kld_weight * kld_loss
        elif args.mmd_weight != 0:
            # MMD term
            if args.prior == 'regular':
                true_samples = torch.randn_like(a, device=self.device)
            elif args.prior == '10mix':
                prior = gaussian_mixture(args.batch_size, args.a_dim)
                true_samples = torch.FloatTensor(prior).to(device=self.device)
            elif args.prior == 'roll':
                prior = swiss_roll(args.batch_size)
                true_samples = torch.FloatTensor(prior).to(device=self.device)
            loss_mmd = compute_mmd(true_samples, a)
            print('mmd loss:', args.mmd_weight * loss_mmd)
            loss += args.mmd_weight * loss_mmd
        elif args.kld_weight != 0:
            # KLD term
            kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
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

        if a is None:
            a, a_q, mu, log_var = self.encoder(x)
        else:
            a_q = a

        if self.mmd_weight != 0 and self.kld_weight != 0:
            output = self.backbone(x_tilde, idx, a_q)
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
        self.is_latent = args.is_latent
        if args.mode == "train_latent_ddim":
            self.is_latent = True
        if args.input_size == 28:
            ch_mult = [1,2,4,]
        else:
            ch_mult = [1,2,4,8]
        if self.is_latent:
            self.backbone = LatentUNet(T=args.diffusion_steps, num_layers=10, dropout=0.1, shape=shape, activation='silu')
        else:
            self.backbone = UNet(ch_mult=ch_mult, T=args.diffusion_steps, ch=args.unets_channels, shape=shape)
        self.to(device)

    def loss_fn(self, args, x, idx=None, curr_epoch=0):
        '''
        x          : real data if idx==None else perturbation data
        idx        : if None (training phase), we perturbed random index.
        '''
        output, epsilon = self.forward(x, idx=idx, get_target=True)
        # denoising matching term
        loss = (output - epsilon).square().mean()
        # print('denoising loss:', loss)
        return loss

    def forward(self, x, idx=None, get_target=False):

        if idx is None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0), )).to(device = self.device)
            if self.is_latent:
                used_alpha_bars = self.alpha_bars[idx][:, None]
            else:
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
        reconstruction, a_q, mu, log_var = self.forward(x, get_target=True)
        # denoising matching term
        loss = (reconstruction - x).square().mean()
        print('reconstruction loss:', loss)
        # prior matching term
        if args.mmd_weight != 0:
            # MMD term
            true_samples = torch.randn_like(a_q, device=self.device)
            loss_mmd = compute_mmd(true_samples, a_q)
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
            reconstruction = self.decoder(a_q)
        elif self.mmd_weight == 0 and self.kld_weight == 0:
            reconstruction = self.decoder(a)
        elif self.mmd_weight != 0:
            reconstruction = self.decoder(a_q)
        elif self.kld_weight != 0:
            reconstruction = self.decoder(a_q)

        return (reconstruction, a_q, mu, log_var) if get_target else reconstruction


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
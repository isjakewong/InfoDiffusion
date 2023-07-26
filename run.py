import argparse
import os
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import get_dataset, get_dataset_config
from models import InfoDiff, Diff, VAE
from sampling import DiffusionProcess, TwoPhaseDiffusionProcess, LatentDiffusionProcess
from utils import (
    AverageMeter, ProgressMeter, GradualWarmupScheduler, \
    generate_exp_string, seed_everything, cos, LatentDataset
)
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--r_seed', type=int, default=0,
                        help='the value of given random seed')
    parser.add_argument('--img_id', type=int, default=0,
                        help='the id of given img')
    parser.add_argument('--model', required=True,
                        choices=['diff', 'vae', 'vanilla'], help='which type of model to run')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'eval', 'eval_fid', 'save_latent', 'disentangle',
                                 'interpolate', 'save_original_img', 'latent_quality', 
                                 'train_latent_ddim', 'plot_latent'], help='which mode to run') 
    parser.add_argument('--prior', required=True,
                        choices=['regular', '10mix', 'roll'], help='which type of prior to run')
    parser.add_argument('--kld_weight', type=float, default=0,
                        help='weight of kld loss')
    parser.add_argument('--mmd_weight', type=float, default=0.1,
                        help='weight of mmd loss')
    parser.add_argument('--use_C', action='store_true',
                        default=False, help='use control constant or not')
    parser.add_argument('--C_max', type=float, default=25,
                        help='control constant of kld loss (orig defualt: 25 for simple, 50 for complex)')
    parser.add_argument('--dataset', required=True,
                        choices=['fmnist', 'mnist', 'celeba', 'cifar10', 'dsprites', 'chairs', 'ffhq'], help='training dataset')
    parser.add_argument('--img_folder', default='./imgs',
                        help='path to save sampled images')
    parser.add_argument('--log_folder', default='./logs',
                        help='path to save logs')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--save_epochs', type=int, default=5,
                        help='number of epochs to save model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--optimizer', default='adam', choices=['adam'],
                        help='optimization algorithm')
    parser.add_argument('--model_folder', default='./models',
                        help='folder where logs will be stored')
    parser.add_argument('--deterministic', action='store_true',
                        default=False, help='deterministid sampling')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='number of input channels')
    parser.add_argument('--unets_channels', type=int, default=64,
                        help='number of input channels')
    parser.add_argument('--encoder_channels', type=int, default=64,
                        help='number of input channels')
    parser.add_argument('--input_size', type=int, default=32,
                        help='expected size of input')
    parser.add_argument('--a_dim', type=int, default=32, required=True,
                        help='dimensionality of auxiliary variable')
    parser.add_argument('--beta1', type=float, default=1e-5,
                        help='value of beta 1')
    parser.add_argument('--betaT', type=float, default=1e-2,
                        help='value of beta T')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='number of diffusion steps')
    parser.add_argument('--split_step', type=int, default=500,
                        help='the step for splitting two phases')
    parser.add_argument('--sampling_number', type=int, default=16,
                        help='number of sampled images')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--tb_logger', action='store_true',
                        help='use tensorboard logger.')
    parser.add_argument('--is_latent', action='store_true',
                        help='use latent diffusion for unconditional sampling.')
    parser.add_argument('--is_bottleneck', action='store_true',
                        help='only fuse aux variable in bottleneck layers.')
    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------------


def save_images(args, sample=None, epoch=0, sample_num=0):
    root = f'{args.img_folder}'
    if args.model == 'vae':
        root = os.path.join(root, 'vae')
    else:
        if args.model == 'vanilla':
            root = os.path.join(root, 'diff')
    root = os.path.join(root, generate_exp_string(args))
    if args.mode == 'eval':
        root = os.path.join(root, 'eval')
    elif args.mode == 'disentangle':
        root = os.path.join(root, f'disentangle-{args.img_id}')
    elif args.mode == 'interpolate':
        root = os.path.join(root, f'interpolate-{args.img_id}')
    elif args.mode == 'save_latent':
        root = os.path.join(root, 'save_latent')
    elif args.mode == 'attr_classification':
        root = os.path.join(root, 'attr_classification')
    elif args.mode == 'plot_latent':
        root = os.path.join(root, 'plot_latent')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'sample-{epoch}.png')

    img_range = (-1, 1)
    if args.mode == 'train':
        save_image(sample, path, normalize=True, range=img_range, nrow=4)
    elif args.mode == 'eval':
        for _ in range(sample_num, sample_num + len(sample)):
            path = os.path.join(root, f"sample{sample_num:05d}.png")
            save_image(sample, path, normalize=True, range=img_range)
    elif args.mode == 'disentangle':
        path = os.path.join(root, f"sample{sample_num}.png")
        save_image(sample, path, normalize=True, range=img_range, nrow=sample.shape[0])
    elif args.mode == 'interpolate':
        path = os.path.join(root, f"sample{sample_num}.png")
        save_image(sample, path, normalize=True, range=img_range, nrow=sample.shape[0])
    elif args.mode == 'plot_latent':
        path = os.path.join(root, f"{args.mode}.png")
        return path
    elif args.mode == 'attr_classification':
        return root

def save_model(args, epoch, model):
    root = f'{args.model_folder}'
    if args.model == 'vae':
        root = os.path.join(root, 'vae')
    else:
        if args.model == 'vanilla':
            root = os.path.join(root, 'diff')
    root = os.path.join(root, generate_exp_string(args))
    if args.mode == "train_latent_ddim":
        root += '_latent' 
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'model-{epoch}.pth')
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch model state to {path}")


def train(args):
    seed_everything(args.r_seed)
    log_dir = f'{args.log_folder}'
    log_dir = os.path.join(log_dir, generate_exp_string(args))    
    tb_logger = SummaryWriter(log_dir=log_dir) if args.tb_logger else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = get_dataset_config(args)
    print(dict(vars(args)))
    dataloader = get_dataset(args)

    if args.model == 'diff':
        model = InfoDiff(args, device, shape)
    elif args.model == 'vanilla':
        model = Diff(args, device, shape)
    elif args.model == 'vae':
        model = VAE(args, device, shape)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(args.epochs, [losses], prefix='Epoch ')

    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=2., warm_epoch=1, after_scheduler=cosineScheduler)

    global_step = 0
    for curr_epoch in trange(0, args.epochs, desc="Epoch #"):
        total_loss = 0
        batch_bar = tqdm(dataloader, desc="Batch #")
        for idx, data in enumerate(batch_bar):
            if args.dataset in ['fmnist', 'mnist', 'celeba', 'cifar10']:
                data = data[0]
            data = data.to(device=device)
            loss = model.loss_fn(args=args, x=data, curr_epoch=curr_epoch)
            batch_bar.set_postfix(loss=format(loss,'.4f'))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item()
            global_step += 1
            if tb_logger:
                tb_logger.add_scalar('train/loss', loss.item(), global_step)
        losses.update(total_loss / idx)
        current_epoch = curr_epoch
        progress.display(current_epoch)
        current_epoch += 1
        warmUpScheduler.step()
        losses.reset()
        if current_epoch % args.save_epochs == 0:
            save_model(args, current_epoch, model)


def eval(args):
    if args.mode != 'train_latent_ddim':
        seed_everything(args.r_seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shape = get_dataset_config(args)
        print(dict(vars(args)))
        root = f'{args.model_folder}'
        if args.model == 'diff':
            model = InfoDiff(args, device, shape)
        elif args.model == 'vanilla':
            model = Diff(args, device, shape)
            root = os.path.join(root, 'diff')
        elif args.model == 'vae':
            model = VAE(args, device, shape)
            root = os.path.join(root, 'vae')
        root = os.path.join(root, generate_exp_string(args))
        path = os.path.join(root, f'model-{args.epochs}.pth')
        print(f"Loading model from {path}")
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        if (args.dataset in ['celeba', 'cifar10', 'mnist', 'fmnist', 'ffhq'] and args.mode in ['eval_fid']):
            if args.is_latent:
                shape_latent = (1, args.a_dim, args.a_dim)
                model2 = Diff(args, device, shape_latent)
                path2 = f'./models/{generate_exp_string(args)}_latent/model-{args.epochs}.pth'
                if os.path.exists(path2):
                    print(f"Loading model from {path2}")
                else:
                    raise FileNotFoundError("The file path {} does not exist, please train the latent diffusion model first.".format(path2))
                model2.load_state_dict(torch.load(path2, map_location=device), strict=True)
            else:
                model2 = Diff(args, device, shape)
                path2 = f'./models/diff/{args.dataset}_{args.a_dim}d/model-{args.epochs}.pth'
                if os.path.exists(path2):
                    print(f"Loading model from {path2}")
                else:
                    raise FileNotFoundError("The file path {} does not exist, please train the vanilla diffusion model first.".format(path2))
                model2.load_state_dict(torch.load(path2, map_location=device), strict=True)
            model2.eval()
        model.eval()
        if args.model in ['diff', 'vanilla']:
            process = DiffusionProcess(args, model, device, shape)
    if args.mode == 'eval':
        if args.model in ['diff', 'vanilla']:
            for sample_num in trange(0, args.sampling_number, args.batch_size, desc="Generating eval images"):
                sample = process.sampling(sampling_number=16)
                save_images(args, sample, sample_num=sample_num)
        elif args.model == 'vae':
            a = torch.randn([args.sampling_number, args.a_dim]).to(device=device)
            sample = model.decoder(a)
            save_images(args, sample)
    elif args.mode == 'eval_fid':
        root = f'{args.img_folder}'
        if args.model == 'vae':
            root = os.path.join(root, 'vae')
        root = os.path.join(root, generate_exp_string(args))
        if args.is_latent:
            root = os.path.join(root, 'eval-fid-latent')
        else:
            root = os.path.join(root, 'eval-fid-fast')
        os.makedirs(root, exist_ok=True)
        print(f"Saving images to {root}")
        if args.model == 'diff':
            if args.is_latent:
                process_latent = LatentDiffusionProcess(args, model2, device)
            else:
                process = TwoPhaseDiffusionProcess(args, model, model2, device, shape)

            for sample_num in trange(0, args.sampling_number, args.batch_size, desc="Generating eval images"):
                if args.is_latent:
                    batch_a = process_latent.sampling(sampling_number=args.batch_size)
                    batch = process.sampling(sampling_number=args.batch_size, a=batch_a)
                else:
                    batch = process.sampling(sampling_number=args.batch_size)
                for batch_num, img in enumerate(batch):
                    img = torch.clip(img, min=-1, max=1)
                    img = ((img + 1)/2) # normalize to 0 - 1
                    img_num = sample_num + batch_num
                    if img_num >= args.sampling_number:
                        return
                    path = os.path.join(root, f'sample-{img_num:06d}.png')
                    save_image(img, path)
            print("DONE")
        elif args.model == 'vae':
            for sample_num in trange(0, args.sampling_number, args.batch_size, desc="Generating eval images"):
                a = torch.randn([args.batch_size, args.a_dim]).to(device=device)
                batch = model.decoder(a)
                for batch_num, img in enumerate(batch):
                    img = torch.clip(img, min=-1, max=1)
                    img = ((img + 1)/2) # normalize to 0 - 1
                    img_num = sample_num + batch_num
                    if img_num >= args.sampling_number:
                        return
                    path = os.path.join(root, f'sample-{img_num:06d}.png')
                    save_image(img, path)
            print("DONE")
    elif args.mode == 'latent_quality':
        process = DiffusionProcess(args, model, device, shape)
        dataloader = get_dataset(args)
        root = f'{args.img_folder}'
        root = os.path.join(root, generate_exp_string(args))
        root = os.path.join(root, 'latent_quality')
        print(f"Saving images to {root}")
        for idx, data in enumerate(dataloader):
            if args.dataset in ['fmnist', 'mnist', 'celeba', 'cifar10', 'dsprites']:
                data_all = data
                data = data_all[0]
            if idx == 10:
                break
        data = data.to(device=device)
        if args.kld_weight != 0:
            with torch.no_grad():
                _, _, mu, log_var = model.encoder(data)
            a = mu + torch.exp(0.5 * log_var)
        elif args.mmd_weight != 0:
            with torch.no_grad():
                a, _, _, _ = model.encoder(data)
        xT = process.reverse_sampling(data, a)
        xT_original = xT.repeat(args.sampling_number, 1, 1, 1)
        a_original = a.repeat(args.sampling_number, 1)
        xT = torch.randn_like(xT_original)
        batch = process.sampling(xT=xT, a=a_original)
        os.makedirs(root, exist_ok=True)
        for batch_num, img in enumerate(batch):
            img = torch.clip(img, min=-1, max=1)
            img = ((img + 1)/2) # normalize to 0 - 1
            path = os.path.join(path, f'sample-{batch_num:06d}.png')
            save_image(img, path)
    elif args.mode == 'plot_latent':
        all_a, all_attr = [], []
        dataloader = get_dataset(args)
        for idx, data in enumerate(dataloader):
            if args.dataset in ['fmnist', 'celeba', 'cifar10', 'dsprites', 'mnist']:
                data_all = data
                data = data_all[0]
                if args.dataset in ['celeba', 'fmnist', 'mnist']:
                    latents_classes = data_all[1]
                elif args.dataset == 'dsprites':
                    latents_classes = data_all[2]
            data = data.to(device=device)
            if (args.mmd_weight == 0 and args.kld_weight == 0):
                with torch.no_grad():
                    a, _, _, _ = model.encoder(data)
            elif (args.mmd_weight != 0):
                with torch.no_grad():
                    a, _, _, _ = model.encoder(data)
            else:
                with torch.no_grad():
                    _, _, mu, _ = model.encoder(data)
                a = mu
            all_a.append(a.cpu().numpy())
            all_attr.append(latents_classes)
        all_a = np.concatenate(all_a)
        all_attr = np.concatenate(all_attr)
        plt.scatter(all_a[:, 0], all_a[:, 1], c = all_attr, cmap = 'tab10', s=5)
        path = save_images(args)
        plt.savefig(path)
    elif args.mode == 'disentangle':
        dataloader = get_dataset(args)
        for idx, data in enumerate(dataloader):
            if args.dataset in ['fmnist', 'mnist', 'celeba', 'cifar10', 'dsprites']:
                data_all = data
                data = data_all[0]
                if args.dataset == 'celeba':
                    latents_classes = data_all[1]
                elif args.dataset == 'dsprites':
                    latents_classes = data_all[2]
            if idx == args.img_id:
                break
        data = data.to(device=device)
        # eta = [-3, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
        eta = [-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
        if args.kld_weight != 0:
            with torch.no_grad():
                _, _, mu, _ = model.encoder(data)
            a = mu
        elif args.mmd_weight != 0:
            with torch.no_grad():
                a, _, _, _ = model.encoder(data)
        if args.model == 'diff':
            xT = process.reverse_sampling(data, a)
            xT = xT.repeat(len(eta), 1, 1, 1)
        for k in range(args.a_dim):
            a_list = []
            for e in eta:
                if args.kld_weight != 0:
                    with torch.no_grad():
                        _, _, mu, log_var = model.encoder(data)
                    a = mu
                    print(mu, log_var)
                elif args.mmd_weight != 0:
                    with torch.no_grad():
                        a, _, _, _ = model.encoder(data)
                a[0][k] = e
                a_list.append(a)
            a = torch.stack(a_list).squeeze(dim=1)
            if args.model == 'diff':
                sample = process.sampling(xT=xT, a=a)
            elif args.model == 'vae':
                sample = model.decoder(a)
            save_images(args, sample, sample_num=k)
    elif args.mode == 'save_latent':
        all_a, all_attr = [], []
        dataloader = get_dataset(args)
        for idx, data in enumerate(dataloader):
            if args.dataset in ['fmnist', 'mnist', 'celeba', 'cifar10', 'dsprites']:
                data_all = data
                data = data_all[0]
                if args.dataset in ['celeba', 'fmnist', 'mnist', 'cifar10']:
                    latents_classes = data_all[1]
                elif args.dataset == 'dsprites':
                    latents_classes = data_all[2]
            else:
                latents_classes = ['No Attributes']
            data = data.to(device=device)
            if args.kld_weight != 0:
                with torch.no_grad():
                    _, _, mu, _ = model.encoder(data)
                a = mu
            elif args.mmd_weight != 0:
                with torch.no_grad():
                    a, _, _, _ = model.encoder(data)
            elif (args.mmd_weight == 0 and args.kld_weight == 0):
                with torch.no_grad():
                    a, _, _, _ = model.encoder(data)
            all_a.append(a.cpu().numpy())
            all_attr.append(latents_classes)
        all_a = np.concatenate(all_a)
        all_attr = np.concatenate(all_attr)
        np.savez("{}_{}_latent".format(args.model, generate_exp_string(args).replace(".", "_")), all_a = all_a, all_attr = all_attr)
    elif args.mode == 'interpolate':
        dataloader = get_dataset(args)
        for idx, data in enumerate(dataloader):
            if args.dataset in ['fmnist', 'mnist', 'celeba', 'cifar10']:
                data = data[0]
            if idx == args.img_id:
                break
        data = data.to(device=device)
        if args.kld_weight != 0:
            with torch.no_grad():
                _, _, mu, _ = model.encoder(data)
            a = mu
        elif args.mmd_weight != 0:
            with torch.no_grad():
                a, _, _, _ = model.encoder(data)
        elif (args.mmd_weight == 0 and args.kld_weight == 0):
            with torch.no_grad():
                a, _, _, _ = model.encoder(data)
        if args.model in ['diff', 'vanilla']:
            xT = process.reverse_sampling(data, a)
            theta = torch.arccos(cos(xT[0], xT[1]))
        a1 = a[0]
        a2 = a[1]
        eta = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1.0]
        intp_a_list = []
        intp_x_list = []
        for e in eta:
            intp_a_list.append(np.cos(e * np.pi / 2) * a1 + np.sin(e * np.pi / 2) * a2)
            if args.model in ['diff', 'vanilla']:
                intp_x = (torch.sin((1 - e) * theta) * xT[0] + torch.sin(e * theta) * xT[1]) / torch.sin(theta)
                intp_x_list.append(intp_x)
        intp_a = torch.stack(intp_a_list)
        if args.model in ['diff', 'vanilla']:
            intp_x = torch.stack(intp_x_list).squeeze(dim=1)
            sample = process.sampling(xT=intp_x, a=intp_a)
        elif args.model == 'vae':
            sample = model.decoder(intp_a)
        save_images(args, sample)
    elif args.mode == 'train_latent_ddim':
        dataset = LatentDataset("{}_{}_latent.npz".format(args.model, generate_exp_string(args).replace(".", "_")))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        seed_everything(args.r_seed)
        log_dir = f'{args.log_folder}'
        log_dir = os.path.join(log_dir, generate_exp_string(args))   
        log_dir += '_latent' 
        tb_logger = SummaryWriter(log_dir=log_dir) if args.tb_logger else None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shape = (1, args.a_dim, args.a_dim)
        model = Diff(args, device, shape)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

        losses = AverageMeter('Loss', ':.4f')
        progress = ProgressMeter(args.epochs, [losses], prefix='Epoch ')

        cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1)
        warmUpScheduler = GradualWarmupScheduler(
            optimizer=optimizer, multiplier=2., warm_epoch=1, after_scheduler=cosineScheduler)

        global_step = 0
        for curr_epoch in trange(0, args.epochs, desc="Epoch #"):
            total_loss = 0
            batch_bar = tqdm(dataloader, desc="Batch #")
            for idx, data in enumerate(batch_bar):
                data = data.to(device=device)
                loss = model.loss_fn(args=args, x=data, curr_epoch=curr_epoch)
                batch_bar.set_postfix(loss=format(loss,'.4f'))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += loss.item()
                global_step += 1
                if tb_logger:
                    tb_logger.add_scalar('train/loss', loss.item(), global_step)
            losses.update(total_loss / idx)
            current_epoch = curr_epoch
            progress.display(current_epoch)
            current_epoch += 1
            warmUpScheduler.step()
            losses.reset()
            if current_epoch % args.save_epochs == 0:
                save_model(args, current_epoch, model)


if __name__ == '__main__':
    args = parse_args()
    if args.mode in ['train']:
        train(args)
    elif args.mode in ['eval', 'eval_fid', 'latent_quality', 'disentangle', 'interpolate', 
                       'save_latent', 'train_latent_ddim', 'plot_latent']:
        if args.mode in ['disentangle', 'latent_quality']:
            args.batch_size = 1
        elif args.mode == 'interpolate':
            args.batch_size = 2
        eval(args)
    elif args.mode in ['save_original_img']:
        from tqdm import tqdm
        import os
        from torchvision.utils import save_image
        output_folder = f'./{args.dataset}_imgs/'
        os.makedirs(output_folder, exist_ok=True)
        dataloader = get_dataset(args)
        for i, img in enumerate(tqdm(dataloader)):
            img = ((img[0] + 1)/2) # normalize to 0 - 1
            save_image(img, f'{output_folder}/{i:06d}.png')
import argparse
import os
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset, get_dataset_config
from models import InfoDiff, Diff, VAE, FeatureClassfier
from sampling import DiffusionProcess, TwoPhaseDiffusionProcess
from utils import (
    AverageMeter, ProgressMeter, GradualWarmupScheduler, \
    generate_exp_string, seed_everything, cos, BCE_loss
)

from metric_utils import DCIMetric, TADMetric
import seaborn as sns
import matplotlib.pyplot as plt

# enable cudnn benchmark
# gives a 1-5% speedup
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', action='store_true',
                        default=False, help='fix random seed')
    parser.add_argument('--model', required=True,
                        choices=['diff', 'vae', 'vanilla'], help='which type of model to run')
    parser.add_argument('--mode', required=True,
                        choices=['train', 'eval', 'eval_fid', 'dis_quant', 'disentangle', \
                        'interpolate', 'attr_classification', 'latent_quality'], help='which mode to run')
    parser.add_argument('--kld_weight', type=float, default=0,  # default=1,
                        help='weight of kld loss (orig default: 1)')
    parser.add_argument('--mmd_weight', type=float, default=0,  # default=0.1
                        help='weight of mmd loss (orig defualt: 0.1)')
    parser.add_argument('--use_C', action='store_true',
                        default=False, help='use control constant or not')
    parser.add_argument('--C_max', type=float, default=25,
                        help='control constant of kld loss (orig defualt: 25 for simple, 50 for complex)')
    parser.add_argument('--dataset', required=True,
                        choices=['fmnist', 'celeba', 'cifar10', 'dsprites', 'chairs'], help='training dataset')
    parser.add_argument('--img_folder', default='./imgs',
                        help='path to save sampled images')
    parser.add_argument('--log_folder', default='./logs',
                        help='path to save logs')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--display_epochs', type=int, default=50,
                        help='number of epochs to display')
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
    parser.add_argument('--a_dim', type=int, default=10, required=True,
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

    parser.add_argument('--disent_metric', default='dci',
                        choices=['dci', 'tad'], help='the disentanglement metric')
    parser.add_argument('--disent_num', default=1000,  type=int, help='the number of the samples to calculate disentanglement metric')

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
        root = os.path.join(root, f'disentangle{epoch}')
    elif args.mode == 'interpolate':
        root = os.path.join(root, 'interpolate')
    elif args.mode == 'dis_quant':
        root = os.path.join(root, 'dis_quant')
    elif args.mode == 'attr_classification':
        root = os.path.join(root, 'attr_classification')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'sample-{epoch}.png')

    # TODO all datasets should be normalized to the same range!
    # if args.dataset in ['fmnist', 'dsprites']:
    #    img_range = (0, 1)
    # else:
    #    img_range = (-1, 1)
    # print(torch.min(sample))
    # print(torch.max(sample))
    # assert torch.all(sample >= img_range[0]) and torch.all(sample <= img_range[1])
    img_range = (-1, 1)
    if args.mode == 'train':
        save_image(sample, path, normalize=True, range=img_range, nrow=4)
    elif args.mode == 'eval':
        for i in range(sample_num, sample_num + len(sample)):
            path = os.path.join(root, f"sample{sample_num:05d}.png")
            save_image(sample, path, normalize=True, range=img_range)
    elif args.mode == 'disentangle':
        path = os.path.join(root, f"sample{sample_num}.png")
        save_image(sample, path, normalize=True, range=img_range, nrow=sample.shape[0])
    elif args.mode == 'interpolate':
        path = os.path.join(root, f"sample{sample_num}.png")
        save_image(sample, path, normalize=True, range=img_range, nrow=sample.shape[0])            
    elif args.mode == 'dis_quant':
        path = os.path.join(root, f"{args.disent_metric}.png")
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
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'model-{epoch}.pth')
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch model state to {path}")


def train(args):
    if args.seed:
        seed_everything()
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
        for idx, data in enumerate(tqdm(dataloader, desc="Batch #")):
            if args.dataset in ['fmnist', 'celeba', 'cifar10']:
                data = data[0]
            data = data.to(device=device)
            loss = model.loss_fn(args=args, x=data, curr_epoch=curr_epoch)
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
        if current_epoch % args.display_epochs == 0:
            save_model(args, current_epoch, model)
            if args.model in ['diff', 'vanilla']:
                process = DiffusionProcess(args, model, device, shape)
                sample = process.sampling()
            elif args.model == 'vae':
                a = torch.randn([args.sampling_number, args.a_dim]).to(device=device)
                sample = model.decoder(a)
            save_images(args, sample, current_epoch)
        losses.reset()


def eval(args):
    if args.seed:
        seed_everything()
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
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    if (args.model in ['diff', 'vanilla'] and args.mode == 'eval_fid'):
        model2 = Diff(args, device, shape)
        model2.load_state_dict(torch.load('./models/diff/celeba_32d_old/model-50.pth', map_location=device), strict=True)
        model2.eval()
    model.eval()
    if args.model in ['diff', 'vanilla']:
        process = DiffusionProcess(args, model, device, shape)
    if args.mode == 'eval':
        # dataloader = get_dataset(args)
        # for idx, data in enumerate(dataloader):
            # if args.dataset in ['fmnist', 'celeba', 'cifar10']:
            #     data = data[0]
            #     data = data.to(device=device)
            # if args.kld_weight != 0:
            #     with torch.no_grad():
            #         _, _, mu, _ = model.encoder(data)
            #     a = mu
            # elif args.mmd_weight != 0:
            #     with torch.no_grad():
            #         a, _, _, _ = model.encoder(data)
        if args.model in ['diff', 'vanilla']:
            for sample_num in trange(0, args.sampling_number, args.batch_size, desc="Generating eval images"):
                sample = process.sampling(sampling_number=16)
                save_images(args, sample, sample_num=sample_num)
        elif args.model == 'vae':
            a = torch.randn([args.sampling_number, args.a_dim]).to(device=device)
            sample = model.decoder(a)
            save_images(args, sample)
            # break
    elif args.mode == 'eval_fid':
        root = f'{args.img_folder}'
        root = os.path.join(root, generate_exp_string(args))
        root = os.path.join(root, 'eval-fid-fast')
        os.makedirs(root, exist_ok=True)
        print(f"Saving images to {root}")
        if args.model == 'diff':
            process = TwoPhaseDiffusionProcess(args, model, model2, device, shape)
            # dataloader = get_dataset(args)
            # for idx, data in enumerate(dataloader):
            #     if args.dataset in ['fmnist', 'celeba', 'cifar10']:
            #         data = data[0]
            #         data = data.to(device=device)
            #     if args.kld_weight != 0:
            #         with torch.no_grad():
            #             _, _, mu, _ = model.encoder(data)
            #         a = mu
            #     elif args.mmd_weight != 0:
            #         with torch.no_grad():
            #             a, _, _, _ = model.encoder(data)
            #     break
            # value_range = (-1, 1)

            for sample_num in trange(0, args.sampling_number, args.batch_size, desc="Generating eval images"):
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
        a = torch.randn_like(a_original)
        # batch1 = process.sampling(xT=xT_original, a=a)
        batch2 = process.sampling(xT=xT, a=a_original)
        # path1 = os.path.join(root, 'xT_fixed')
        path2 = os.path.join(root, 'a_fixed')
        # os.makedirs(path1, exist_ok=True)
        os.makedirs(path2, exist_ok=True)
        # for batch_num, img in enumerate(batch1):
        #     img = torch.clip(img, min=-1, max=1)
        #     img = ((img + 1)/2) # normalize to 0 - 1
        #     path = os.path.join(path1, f'sample-{batch_num:06d}.png')
        #     save_image(img, path)
        for batch_num, img in enumerate(batch2):
            img = torch.clip(img, min=-1, max=1)
            img = ((img + 1)/2) # normalize to 0 - 1
            path = os.path.join(path2, f'sample-{batch_num:06d}.png')
            save_image(img, path)
    elif args.mode == 'disentangle':
        dataloader = get_dataset(args)
        for idx, data in enumerate(dataloader):
            if args.dataset in ['fmnist', 'celeba', 'cifar10', 'dsprites']:
                data_all = data
                data = data_all[0]
                if args.dataset == 'celeba':
                    all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                                'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
                                'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
                                'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
                                'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
                                'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                                'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' 
                    ]
                    latents_classes = data_all[1]
                elif args.dataset == 'dsprites':
                    all_attrs = ['Color', 'Shape', 'Scale', 'Orientation', 'Position_X', 'Position_Y']
                    latents_values = data_all[1]
                    latents_classes = data_all[2]
            if idx==92:
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
            # print(a)
            if args.model == 'diff':
                sample = process.sampling(xT=xT, a=a)
            elif args.model == 'vae':
                sample = model.decoder(a)
            save_images(args, sample, epoch=idx, sample_num=k)
    elif args.mode == 'dis_quant':
        all_a, all_attr = [], []
        dataloader = get_dataset(args)
        for idx, data in enumerate(dataloader):
            if idx >= args.disent_num:
                break
            if args.dataset in ['fmnist', 'celeba', 'cifar10', 'dsprites']:
                data_all = data
                data = data_all[0]
                if args.dataset == 'celeba':
                    all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                                'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
                                'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
                                'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
                                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
                                'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
                                'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                                'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' 
                    ]
                    latents_classes = data_all[1]
                elif args.dataset == 'dsprites':
                    all_attrs = ['Color', 'Shape', 'Scale', 'Orientation', 'Position_X', 'Position_Y']
                    latents_values = data_all[1]
                    latents_classes = data_all[2]

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
            print(idx, a)

        all_a = np.concatenate(all_a)
        all_attr = np.concatenate(all_attr)

        np.savez("{}_{}_disentangle".format(args.model, generate_exp_string(args).replace(".", "_")), all_a = all_a, all_attr = all_attr)

        # if args.disent_metric == "dci":
        #     if 'dsprites' in args.dataset:
        #         dci_metric = DCIMetric("RandomForestIBGAN")
        #     else:
        #         dci_metric = DCIMetric("RandomForestCV")
        #     dci_result = dci_metric.evaluate(all_a, all_attr)

        #     R = dci_result['DCI_RandomForestCV_metric_detail']

        #     if 'dsprites' in args.dataset:
        #         print("DCI Score", dci_result['DCI_RandomForestIBGAN_disent_metric'])
        #     else:
        #         print("DCI Score", dci_result['DCI_RandomForestCV_disent_metric'])

        #     print("Plot: relative importance")
        #     sns.heatmap(R, xticklabels=all_attrs)
        #     path = save_images(args)
        #     plt.savefig(path)
        # else:
        #     tad_metric = TADMetric(all_attr.shape[1], all_attrs)
        #     tad_score, auroc_result, num_attr = tad_metric.evaluate(all_a, all_attr)

        #     print("TAD SCORE: ", tad_score, "Attributes Captured: ", num_attr)
        #     sns.heatmap(auroc_result.transpose(), xticklabels=all_attrs)
        #     path = save_images(args)
        #     plt.savefig(path)
    elif args.mode == 'interpolate':
        dataloader = get_dataset(args)
        for idx, data in enumerate(dataloader):
            if args.dataset in ['fmnist', 'celeba', 'cifar10']:
                data = data[0]
            # if idx==15:
            #     break
            data = data.to(device=device)
            if args.kld_weight != 0:
                with torch.no_grad():
                    _, _, mu, _ = model.encoder(data)
                a = mu
            elif args.mmd_weight != 0:
                with torch.no_grad():
                    a, _, _, _ = model.encoder(data)
            if args.model == 'diff':
                xT = process.reverse_sampling(data, a)
                theta = torch.arccos(cos(xT[0], xT[1]))
            a1 = a[0]
            a2 = a[1]
            eta = [0.0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 1.0]
            intp_a_list = []
            intp_x_list = []
            for e in eta:
                intp_a_list.append(np.cos(e * np.pi / 2) * a1 + np.sin(e * np.pi / 2) * a2)
                if args.model == 'diff':
                    intp_x = (torch.sin((1 - e) * theta) * xT[0] + torch.sin(e * theta) * xT[1]) / torch.sin(theta)
                    intp_x_list.append(intp_x)
            intp_a = torch.stack(intp_a_list)
            if args.model == 'diff':
                intp_x = torch.stack(intp_x_list).squeeze(dim=1)
                sample = process.sampling(xT=intp_x, a=intp_a)
            elif args.model == 'vae':
                sample = model.decoder(intp_a)
            save_images(args, sample, sample_num=idx)
    elif args.mode == 'attr_classification':
        train_loader, valid_loader, test_loader = get_dataset(args)
        if args.dataset == 'celeba':
            all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                        'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
                        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
                        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
                        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
                        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
                        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                        'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' 
            ]
            # To be optimized
            attr_nums = [i for i in range(len(all_attrs))] 
            attr_loss_weight = [1 for i in range(len(all_attrs))]
            attr_loss_weight[0] = 10  # attractive 
            attr_loss_weight[5] = 5  # brown_hair
            attr_threshold = [0.5 for i in range(len(all_attrs))]  
        elif args.dataset == 'dsprites':
            all_attrs = ['Color', 'Shape', 'Scale', 'Orientation', 'Position_X', 'Position_Y']
        classifier = FeatureClassfier(args, len(all_attrs))
        optim_ = torch.optim.Adam(classifier.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_, [30,80], gamma=0.1)
        best_valid_acc = 0
        for epoch in range(args.epochs):
            temp_loss = 0.0
            for batch_idx, data in enumerate(dataloader):
                scheduler.step()
                if args.dataset in ['fmnist', 'celeba', 'cifar10']:
                    data_all = data
                    data = data_all[0]
                    if args.dataset == 'celeba':
                        latents_classes = data_all[1]
                    elif args.dataset == 'dsprites':
                        latents_values = data_all[1]
                        latents_classes = data_all[2]
                data = data.to(device=device)
                latents_classes = latents_classes.to(device=device)
                if args.kld_weight != 0:
                    with torch.no_grad():
                        _, _, mu, _ = model.encoder(data)
                    a = mu
                elif args.mmd_weight != 0:
                    with torch.no_grad():
                        a, _, _, _ = model.encoder(data)
                outputs = classifier(a)
                loss = BCE_loss(outputs, latents_classes, attr_loss_weight)
                optim_.zero_grad()
                loss.backward()
                optim_.step()
                temp_loss += loss.item()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(a), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), temp_loss/100))
                    temp_loss = 0.0

            if epoch % 10 == 0:
                path = save_images(args)
                path = os.path.join(path, f"{epoch}.pth")
                torch.save(classifier.state_dict(), path)

            model.eval()
            with torch.no_grad():
                for batch_idx, samples in enumerate(valid_loader):
                    if args.dataset in ['fmnist', 'celeba', 'cifar10']:
                        data_all = data
                        data = data_all[0]
                        if args.dataset == 'celeba':
                            latents_classes = data_all[1]
                        elif args.dataset == 'dsprites':
                            latents_values = data_all[1]
                            latents_classes = data_all[2]
                    data = data.to(device=device)
                    latents_classes = latents_classes.to(device=device)
                    if args.kld_weight != 0:
                        with torch.no_grad():
                            _, _, mu, _ = model.encoder(data)
                        a = mu
                    elif args.mmd_weight != 0:
                        with torch.no_grad():
                            a, _, _, _ = model.encoder(data)
                    labels = latents_classes.tolist()
                    outputs = classifier(a)

                    for i in range(128):
                        for j, attr in enumerate(all_attrs):
                            pred = outputs[i].data[j]
                            pred = 1 if pred > attr_threshold[j] else 0

                            # record accuracy
                            if pred == labels[i][j]:
                                correct_dict[attr] = correct_dict[attr] + 1

                    if batch_idx % 50 == 0:
                        print("Batch_idx : {}/{}".format( 
                                    batch_idx, int(len(valid_loader.dataset)/128)))

                i = 0
                # get the average accuracy
                for attr in all_attrs:
                    correct_dict[attr] = correct_dict[attr] * 100 / len(valid_loader.dataset)                                                                   
                    i += 1
                
                mean_attributes_acc = 0.0
                for k, v in correct_dict.items():
                    mean_attributes_acc += v
                mean_attributes_acc /= len(all_attrs)

                if mean_attributes_acc > best_valid_acc:
                    best_valid_acc = mean_attributes_acc
                    path = save_images(args)
                    path = os.path.join(path, "best.pth")
                    torch.save(model.state_dict(), path)

if __name__ == '__main__':
    args = parse_args()

    # TODO save args dict and add better logging
    if args.mode in ['train']:
        train(args)
    elif args.mode in ['eval', 'eval_fid', 'disentangle', 'interpolate', 'dis_quant', 'attr_classification', 'latent_quality']:
        if args.mode == 'disentangle':
            args.batch_size = 1
        elif args.mode == 'interpolate':
            args.batch_size = 2
        eval(args)

    # from tqdm import tqdm
    # import os
    # from torchvision.utils import save_image
    # output_folder = './fmnist_imgs/'
    # os.makedirs(output_folder, exist_ok=True)
    # dataloader = get_dataset(args)
    # for i, img in enumerate(tqdm(dataloader)):
    #     img = ((img[0] + 1)/2) # normalize to 0 - 1
    #     save_image(img, f'{output_folder}/{i:06d}.png')
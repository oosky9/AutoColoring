import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils import load_datasets
from model import Generator, Discriminator

import statistics
import os
import argparse
import glob
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pre_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--LAMBDA", type=float, default=100)
    parser.add_argument("--train_data", type=str, default="D:/Data/anime/train/")
    parser.add_argument("--valid_data", type=str, default="D:/data/anime/test/")
    parser.add_argument("--pre_image_path", type=str, default="./pre_image/")
    parser.add_argument("--pre_model_path", type=str, default="./pre_model/")
    parser.add_argument("--image_path", type=str, default="./anime_image/")
    parser.add_argument("--model_path", type=str, default="./anime_model/")
    parser.add_argument("--test_image_path", type=str, default="./test_image/")
    parser.add_argument("--mode", type=str, default="pre_train", help=["pre_train", "train", "test"])

    args = parser.parse_args()
    return args

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def get_model_name(p):
    model_name = glob.glob(os.path.join(p, "gen_*.pt"))[-1]
    return model_name


def pre_train(args):

    writer = SummaryWriter()

    model_G = Generator()
    model_G = nn.DataParallel(model_G)
    model_G = model_G.to(device)

    optim_G = torch.optim.Adam(model_G.parameters(),
                               lr=0.0002, betas=(0.5, 0.999))

    loss_G = nn.L1Loss().to(device)

    result = {}
    result["train/mae_loss"] = []
    result["valid/mae_loss"] = []

    train_dataset = load_datasets(args.train_data, args.batch_size, args.patch_size, True)
    valid_dataset = load_datasets(args.valid_data, args.batch_size, args.patch_size, True)

    for i in range(args.pre_epochs):
        mae_loss = []

        for (real_color, input_gray, hint_color), _ in tqdm(train_dataset):

            real_color = real_color.to(device)
            input_gray = input_gray.to(device)
            hint_color = hint_color.to(device)

            optim_G.zero_grad()

            fake_color = model_G(input_gray, hint_color)

            g_loss = loss_G(fake_color, real_color)
            g_loss.backward()

            optim_G.step()

            mae_loss.append(g_loss.item())

        result["train/mae_loss"].append(statistics.mean(mae_loss))

        writer.add_scalar("pre_train/train/mae_loss", result["train/mae_loss"][-1], i + 1)


        if (i + 1) % 1 == 0 or (i + 1) == args.epochs or i == 0:
            with torch.no_grad():
                mae_loss = []
                for (real_color, input_gray, hint_color), _ in tqdm(valid_dataset):
                    batch_len = len(real_color)

                    real_color = real_color.to(device)
                    input_gray = input_gray.to(device)
                    hint_color = hint_color.to(device)

                    fake_color = model_G(input_gray, hint_color)

                    g_loss = loss_G(fake_color, real_color)

                    mae_loss.append(g_loss.item())

                result["valid/mae_loss"].append(statistics.mean(mae_loss))

                writer.add_scalar("pre_train/valid/mae_loss", result["valid/mae_loss"][-1], i + 1)

                torchvision.utils.save_image(real_color[:min(batch_len, 50)],
                                             os.path.join(args.pre_image_path, f"real_epoch_{i + 1:03}.png"),
                                             nrow=5, range=(-1.0, 1.0), normalize=True)
                torchvision.utils.save_image(fake_color[:min(batch_len, 50)],
                                             os.path.join(args.pre_image_path, f"fake_epoch_{i + 1:03}.png"),
                                             nrow=5, range=(-1.0, 1.0), normalize=True)
                torchvision.utils.save_image(hint_color[:min(batch_len, 50)],
                                             os.path.join(args.pre_image_path, f"hint_epoch_{i + 1:03}.png"),
                                             nrow=5, range=(-1.0, 1.0), normalize=True)

                torch.save(model_G.state_dict(),
                           os.path.join(args.pre_model_path, f"gen_{i + 1:03}.pt"))

    writer.close()


def train(args):

    writer = SummaryWriter()

    model_name = get_model_name(args.pre_model_path)
    print("loading...{}".format(model_name))

    model_G, model_D = Generator(), Discriminator(args.patch_size)
    model_G, model_D = nn.DataParallel(model_G), nn.DataParallel(model_D)
    model_G.load_state_dict(torch.load(model_name))
    model_G, model_D = model_G.to(device), model_D.to(device)

    optim_G = torch.optim.Adam(model_G.parameters(),
                                lr=0.0002, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(model_D.parameters(),
                                lr=0.0002, betas=(0.5, 0.999))

    loss_bce = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()

    ones = torch.ones(512, 1, args.patch_size//16, args.patch_size//16).to(device)
    zeros = torch.zeros(512, 1, args.patch_size//16, args.patch_size//16).to(device)

    result = {}
    result["train/total_loss_G"] = []
    result["train/total_loss_D"] = []
    result["valid/total_loss_G"] = []
    result["valid/total_loss_D"] = []

    train_dataset = load_datasets(args.train_data, args.batch_size, args.patch_size, True)
    valid_dataset = load_datasets(args.valid_data, args.batch_size, args.patch_size, True)

    for i in range(args.epochs):
        total_loss_G, total_loss_D = [], []

        for (real_color, input_gray, hint_color), _ in tqdm(train_dataset):
            batch_len = len(real_color)

            real_color = real_color.to(device)
            input_gray = input_gray.to(device)
            hint_color = hint_color.to(device)

            optim_D.zero_grad()
            optim_G.zero_grad()

            fake_color = model_G(input_gray, hint_color)
            fake_color_tensor = fake_color.detach()

            fake_D = model_D(fake_color)

            g_mse_loss = loss_mse(real_color, fake_color)
            g_bce_loss = loss_bce(fake_D, ones[:batch_len])
            g_loss = args.LAMBDA * g_mse_loss + g_bce_loss
            g_loss.backward(retain_graph=True)
            optim_G.step()

            total_loss_G.append(g_loss.item())

            real_D_out = model_D(real_color)
            fake_D_out = model_D(fake_color_tensor)

            d_real_loss = loss_bce(real_D_out, ones[:batch_len])
            d_fake_loss = loss_bce(fake_D_out, zeros[:batch_len])
            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            optim_D.step()

            total_loss_D.append(d_loss.item())

        result["train/total_loss_G"].append(statistics.mean(total_loss_G))
        result["train/total_loss_D"].append(statistics.mean(total_loss_D))

        writer.add_scalar('train/loss_G', result['train/total_loss_G'][-1], i + 1)
        writer.add_scalar('train/loss_D', result['train/total_loss_D'][-1], i + 1)

        if (i + 1) % 1 == 0 or (i + 1) == args.epochs or i == 0:

            with torch.no_grad():
                total_loss_G, total_loss_D = [], []
                for (real_color, input_gray, hint_color), _ in tqdm(valid_dataset):
                    batch_len = len(real_color)

                    real_color = real_color.to(device)
                    input_gray = input_gray.to(device)
                    hint_color = hint_color.to(device)

                    fake_color = model_G(input_gray, hint_color)
                    fake_color_tensor = fake_color.detach()

                    fake_D = model_D(fake_color)

                    g_mse_loss = loss_mse(real_color, fake_color)
                    g_bce_loss = loss_bce(fake_D, ones[:batch_len])
                    g_loss = args.LAMBDA * g_mse_loss + g_bce_loss

                    total_loss_G.append(g_loss.item())

                    real_D_out = model_D(real_color)
                    fake_D_out = model_D(fake_color_tensor)

                    d_real_loss = loss_bce(real_D_out, ones[:batch_len])
                    d_fake_loss = loss_bce(fake_D_out, zeros[:batch_len])
                    d_loss = d_real_loss + d_fake_loss

                    total_loss_D.append(d_loss.item())

                result["valid/total_loss_G"].append(statistics.mean(total_loss_G))
                result["valid/total_loss_D"].append(statistics.mean(total_loss_D))

                writer.add_scalar('valid/loss_G', result['valid/total_loss_G'][-1], i + 1)
                writer.add_scalar('valid/loss_D', result['valid/total_loss_D'][-1], i + 1)

                torchvision.utils.save_image(real_color[:min(batch_len, 50)],
                                             os.path.join(args.image_path, f"real_epoch_{i + 1:03}.png"),
                                             nrow=5, range=(-1.0, 1.0), normalize=True)
                torchvision.utils.save_image(fake_color[:min(batch_len, 50)],
                                             os.path.join(args.image_path, f"fake_epoch_{i + 1:03}.png"),
                                             nrow=5, range=(-1.0, 1.0), normalize=True)
                torchvision.utils.save_image(hint_color[:min(batch_len, 50)],
                                             os.path.join(args.image_path, f"hint_epoch_{i + 1:03}.png"),
                                             nrow=5, range=(-1.0, 1.0), normalize=True)

                torch.save(model_G.state_dict(),
                           os.path.join(args.model_path, f"gen_{i + 1:03}.pt"))
                torch.save(model_D.state_dict(),
                           os.path.join(args.model_path, f"dis_{i + 1:03}.pt"))

    writer.close()

def test(args):

    model_name = get_model_name("./anime_model/")
    print("loading...{}".format(model_name))

    model_G = Generator()
    model_G = nn.DataParallel(model_G)
    model_G.load_state_dict(torch.load(model_name))
    model_G = model_G.to(device)

    model_G.eval()

    valid_dataset = load_datasets(args.valid_data, 50, args.patch_size, False)

    i = 0
    with torch.no_grad():
        for (real, img, hint), _ in valid_dataset:
            i += 1
            batch_len = len(img)
            img = img.to(device)
            hint = hint.to(device)
            fake = model_G(img, hint)

            torchvision.utils.save_image(fake[:min(batch_len, 50)],
                                         os.path.join(args.test_image_path, f"./output_{i:03}.png"),
                                         nrow=10, range=(-1.0, 1.0), normalize=True)


def coloring():
    from PIL import Image
    import numpy as np

    img = Image.open("D:/Data/anime_images/linear/img001.jpg")
    hint = Image.open("D:/Data/anime_images/linear/hint001.jpg")

    img = np.array(img).astype("float32")
    hint = np.array(hint).astype("float32")

    img = img / 127.5 - 1
    hint = hint / 127.5 - 1

    img = img.transpose([2, 0, 1])
    hint = hint.transpose([2, 0, 1])

    img = torch.Tensor(img).to(device)
    hint = torch.Tensor(hint).to(device)

    img = img.unsqueeze(0)
    hint = hint.unsqueeze(0)

    model_name = get_model_name("./anime_model/")
    print("loading...{}".format(model_name))


    model_G = Generator()
    model_G = nn.DataParallel(model_G)
    model_G.load_state_dict(torch.load(model_name))
    model_G = model_G.to(device)

    model_G.eval()

    fake = model_G(img, hint)

    torchvision.utils.save_image(fake[:min(1, 50)],
                                 f"./output.png",
                                 nrow=10, range=(-1.0, 1.0), normalize=True)

def main(args):
    if args.mode == "pre_train":
        check_dir(args.pre_image_path)
        check_dir(args.pre_model_path)
        pre_train(args)
    elif args.mode == "train":
        check_dir(args.image_path)
        check_dir(args.model_path)
        train(args)
    elif args.mode == "test":
        check_dir(args.test_image_path)
        test(args)
    else:
        coloring()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = arg_parser()

    main(args)

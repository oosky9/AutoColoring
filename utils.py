import torch
import torchvision
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageOps
import numpy as np

class ColorAndLinearAndHint(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        img = img.resize((self.patch_size, self.patch_size))
        real, inp, hint = self.get_masked_hints(img)
        real, inp, hint = self.normalize(real, inp, hint)
        return real, inp, hint

    def get_linear(self, image):
        gray = image.convert("L")
        gray2 = gray.filter(ImageFilter.MaxFilter(5))
        senga_inv = ImageChops.difference(gray, gray2)
        senga = ImageOps.invert(senga_inv).convert("RGB")
        return senga

    def normalize(self, real_image, input_image, hint_image):
        real_image = (real_image / 127.5) - 1
        input_image = (input_image / 127.5) - 1
        hint_image = (hint_image / 127.5) - 1

        return real_image, input_image, hint_image

    def get_masked_hints(self, image):
        senga = self.get_linear(image)
        senga = np.array(senga).astype("float32")
        senga_ = 255. - senga

        image = np.array(image).astype("float32")

        mask = Image.new('RGB', (self.patch_size, self.patch_size), (255, 255, 255))
        draw = ImageDraw.Draw(mask)

        for i in range(30):
            x = np.random.randint(0, self.patch_size)
            y = np.random.randint(0, self.patch_size)
            d_x = np.random.randint(-30, 30)
            d_y = np.random.randint(-30, 30)

            draw.line(((x, y), (x + d_x, y + d_y)), fill=(0, 0, 0), width=5)
        mask = np.array(mask).astype("float32")

        hints = image + mask
        hints = np.clip(hints, 0., 255.)
        hints = hints - senga_
        hints = np.clip(hints, 0., 255.)

        return image, senga, hints


class MultiInputWrapper(object):
    def __init__(self, base_func):
        self.base_func = base_func

    def __call__(self, xs):
        if isinstance(self.base_func, list):
            return [f(x) for f, x in zip(self.base_func, xs)]
        else:
            return [self.base_func(x) for x in xs]

def load_datasets(data_path, batch_size, patch_size, shuffle):
    transform = transforms.Compose([
        ColorAndLinearAndHint(patch_size),
        MultiInputWrapper(transforms.ToTensor())
    ])
    images = datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def debug():
    data = load_datasets("D:/Data/anime/train/", batch_size=128, patch_size=256, shuffle=True)

    for (image, senga, hints), _ in data:
        torchvision.utils.save_image(image[:20],
                                     f"image.png",
                                     range=(-1.0, 1.0), normalize=True)
        torchvision.utils.save_image(senga[:20],
                                     f"senga.png",
                                     range=(-1.0, 1.0), normalize=True)
        torchvision.utils.save_image(hints[:20],
                                     f"hints.png",
                                     range=(-1.0, 1.0), normalize=True)
        break

if __name__ == "__main__":
    debug()

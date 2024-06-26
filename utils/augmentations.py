import math
import numbers
import random
import numpy as np
from PIL import Image, ImageOps


class RandomHorizontallyFlip_np:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return np.fliplr(img), np.fliplr(mask)
        return img, mask

class RandomVerticallyFlip_np:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return np.flipud(img), np.flipud(mask)
        return img, mask

class AddNoise_np:
    def __call__(self, img, mask):
        noise = np.random.normal(0, 0.02, img.shape)  # Make sure noise shape matches image shape
        return img + noise, mask

class RandomRotate_np:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.uniform(-self.degree, self.degree)
        img = self.rotate_image(img, rotate_degree)
        mask = self.rotate_image(mask, rotate_degree, mode='nearest')
        return img, mask

    def rotate_image(self, img, angle, mode='bilinear'):
        # For simplicity, using scipy for rotation
        from scipy.ndimage import rotate
        return rotate(img, angle, reshape=False, order=1 if mode == 'bilinear' else 0)

# Define the augmentation class using PIL on images
class DataAugmentationImage:
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        img = Image.fromarray(img, mode='L')
        mask = Image.fromarray(mask, mode='L')
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask)
    
class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):

        img, mask = Image.fromarray(img, mode=None), Image.fromarray(mask, mode="L")
        assert img.size == mask.size

        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)


class AddNoise(object):
    def __call__(self, img, mask):
        noise = np.random.normal(loc=0, scale=0.02, size=(img.size[1], img.size[0]))
        return img + noise, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            # Note: we use FLIP_TOP_BOTTOM here intentionaly. Due to the dimensions of the image,
            # it ends up being a horizontal flip.
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class RandomVerticallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            img.resize(self.size, Image.BILINEAR),
            mask.resize(self.size, Image.NEAREST),
        )


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        """
        PIL automatically adds zeros to the borders of images that rotated. To fix this
        issue, the code in the botton sets anywhere in the labels (mask) that is zero to
        255 (the value used for ignore_index).
        """
        rotate_degree = random.random() * 2 * self.degree - self.degree

        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        binary_mask = Image.fromarray(np.ones([mask.size[1], mask.size[0]]))
        binary_mask = binary_mask.rotate(rotate_degree, Image.NEAREST)
        binary_mask = np.array(binary_mask)

        mask_arr = np.array(mask)
        mask_arr[binary_mask == 0] = 255
        mask = Image.fromarray(mask_arr)

        return img, mask


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )

        return self.crop(*self.scale(img, mask))
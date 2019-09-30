import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Iterable

from PIL import Image, ImageFont, ImageDraw
import numpy as np

def save_image_to_file(image, path: Union[Path, str]):
    if not path.parent.exists():
        Path.mkdir(path.parent, parents=True)

    image = convert_to_PIL(image)
    image.save(str(path))


def show_numpy_img_pil(array):
    img = Image.fromarray(array, 'RGB')
    img.show()


def show_numpy_img_sxiv(array):
    img = Image.fromarray(array, 'RGB')
    tmp_path = tempfile.mktemp(suffix='img.jpg')
    # print(tmp_path)
    img.save(tmp_path)
    viewer = subprocess.Popen(['sxiv', tmp_path])
    viewer.wait()


def show_PIL_img_in_sxiv(img):
    show_numpy_img_sxiv(PIL_to_ndarray(img))


def show_numpy_img_plot(array):
    from matplotlib import pyplot as plt
    plt.imshow(array, interpolation='nearest')
    plt.show()


#### Drawing stuff on images ####

_font = None
DEFAULT_FONT_SIZE = 20


def get_default_font():
    global _font
    if _font is None:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/fonts/Xerox Serif Narrow.ttf")
        _font = create_font(path, DEFAULT_FONT_SIZE)
    return _font


def get_default_font_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/fonts/Xerox Serif Narrow.ttf")


def create_font(path, size):
    return ImageFont.truetype(path, size=size)


def ndarray_to_PIL(img: np.ndarray):
    return Image.fromarray(np.ndarray.astype(img, np.uint8))


def PIL_to_ndarray(img: Image.Image):
    return np.asarray(img)


def convert_to_PIL(img: Union[np.ndarray, Image.Image]):
    if isinstance(img, np.ndarray):
        img = ndarray_to_PIL(img)

    return img

def convert_to_ndarray(img: Union[np.ndarray, Image.Image]):
    if isinstance(img, Image.Image):
        img = PIL_to_ndarray(img)
    return img


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def write_text_on_image(img, text: Union[str, Iterable[str]], font=None, positon=(0, 0), color=BLACK):
    font = font or get_default_font()
    img = convert_to_PIL(img)

    if not isinstance(text, str):
        text = '\n'.join(text)

    if text == '':
        return img

    draw = ImageDraw.Draw(img)
    draw.text(positon, text, color, font=font)

    return img


def concatenate_images_with_resize(imgs, axis=1):
    assert axis in [0, 1]
    imgs = [convert_to_ndarray(img) for img in imgs]
    shapes = np.stack(map(lambda img: img.shape, imgs))

    dim0_resize, dim1_resize = None, None
    if axis == 0:
        dim1_resize = np.max(shapes[:, 1])
    elif axis == 1:
        dim0_resize = np.max(shapes[:, 0])

    imgs_resized = []
    for img in imgs:
        dim0 = dim0_resize or img.shape[0]
        dim1 = dim1_resize or img.shape[1]
        new_img = np.zeros((dim0, dim1, 3), dtype=img.dtype)
        new_img[0:img.shape[0], 0:img.shape[1], :] = np.copy(img)
        imgs_resized.append(new_img)
    return concatenate_images(imgs_resized, axis)



def concatenate_images(imgs, axis=1):
    # See example below
    imgs = [convert_to_PIL(img) for img in imgs]
    imgs_np = [np.array(img) for img in imgs]

    # for img in imgs_np:
    #     print(img.shape)
    concatenated_im_np = np.concatenate(imgs_np, axis=axis)

    return ndarray_to_PIL(concatenated_im_np)

def add_text_bottom(img, text, text_img_size=(300, 1024)):
    h, w, _ = img.shape
    assert img.dtype == 'uint8'
    text_img = np.ones(text_img_size + (3,), dtype='uint8') * 255
    text_img = write_text_on_image(text_img, text)
    img = PIL_to_ndarray(concatenate_images_with_resize([img, text_img], axis=0))
    return img

# Example code for writing on images
# img1 = np.zeros((100, 100, 3), dtype=np.uint8)
# img1[..., 2] = 255
# img2 = np.zeros((100, 100, 3), dtype=np.uint8)
# img2[..., 1] = 255
#
# img1 = write_on_image(img1, "I love RL")
# img2 = write_on_image(img2, "AWARE rules!")
#
# img_3 = concatenate_images(img1, img2)
#
# img_3.save("/tmp/test.png")%

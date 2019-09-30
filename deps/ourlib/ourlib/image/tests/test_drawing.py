import numpy as np

from ourlib.image.utils import concatenate_images, show_numpy_img_sxiv, PIL_to_ndarray, write_text_on_image, \
    show_PIL_img_in_sxiv, BLACK, get_default_font_path, create_font, add_text_bottom


def test_concatenate_images():
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img1[..., 2] = 255

    img2 = np.zeros((200, 100, 3), dtype=np.uint8)
    img2[..., 1] = 255

    img_3 = concatenate_images(img1, img2, axis=1)
    assert PIL_to_ndarray(img_3).shape == (200, 300, 3)
    show_numpy_img_sxiv(PIL_to_ndarray(img_3))



def test_write_on_image():
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img1[..., 2] = 255

    white_img = np.zeros((200, 200, 3), dtype=np.uint8)
    white_img[..., :] = 255
    font = create_font(get_default_font_path(), size=10)
    img = white_img
    img = write_text_on_image(img, 'reward = 1000.0\nreward = 20000.0', color=BLACK, font=font)
    img = write_text_on_image(img, ['reward = 1000.0', 'reward = 20000.0'], color=BLACK, font=font)
    show_PIL_img_in_sxiv(img)


def test_add_text_bottom():
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[..., 2] = 255
    t1 = 'q_values = [11111111, 1111111111, 1111111111111, 111111111]'
    text = [t1, 'reward = 1000.0', 'reward = 20000.0']
    img_with_text = add_text_bottom(img1, text)
    show_PIL_img_in_sxiv(img_with_text)




if __name__ == '__main__':
    # test_concatenate_images()
    # test_write_on_image()
    test_add_text_bottom()

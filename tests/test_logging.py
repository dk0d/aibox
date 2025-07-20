# %%
# import torch

# from aibox.torch.callbacks import LogImagesCallback
# from aibox.torch.image import interlace_images
# from aibox.torch.logging import CombinedLogger

DEFAULT_CONFIG_DIR = "tests/resources/configs"


# def test_cli_no_args():
#     logger = CombinedLogger("./logs")


# def test_log_images_callback_interlace():
#     nImages = 7  # cols (number of images to interlace)
#     maxImages = 3  # rows (max images per batch to show)
#     images = [torch.ones((10, 3, 32, 32)) * i for i in range(1, nImages + 1)]
#     image = interlace_images(images, maxImages=maxImages)
#     assert image.shape == (nImages * maxImages, 3, 32, 32), f"Expected shape (3*4, 3, 32, 32), got {image.shape}"
#     assert (
#         image[1].shape == images[1][0].shape
#     ), f"Expected first images shape to be same, got {image[0].shape} and {images[0][0].shape}"
#     assert torch.equal(image[1], images[1][0]), f"Expected image[1] == images[1], got {image[1]} != {images[1][0]}"

# grid = torchvision.utils.make_grid(image, nrow=nImages, padding=1)
# grid = np.array(ToPILImage()(grid).convert("L"))
# grid = (label2rgb(grid, grid) * 255).astype(np.uint8)
# grid = ToPILImage()(grid)
# display_images([grid])

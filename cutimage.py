import os
from PIL import Image

def split_large_images(input_folder, output_folder, tile_size=(512, 512)):
    """
    Split large RGB images in the input folder and save them into the output folder as tiles.

    Parameters:
        input_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where the split images will be saved.
        tile_size (tuple): Size of the tiles (default is (512, 512) pixels).

    Returns:
        None
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are JPEG or PNG format
            file_path = os.path.join(input_folder, filename)
            # Open image
            img = Image.open(file_path)
            width, height = img.size
            # Calculate number of tiles in horizontal and vertical directions
            num_horizontal = (width + tile_size[0] - 1) // tile_size[0]
            num_vertical = (height + tile_size[1] - 1) // tile_size[1]
            # Split the image into tiles
            for i in range(num_horizontal):
                for j in range(num_vertical):
                    left = i * tile_size[0]
                    upper = j * tile_size[1]
                    right = min((i + 1) * tile_size[0], width)
                    lower = min((j + 1) * tile_size[1], height)
                    box = (left, upper, right, lower)
                    tile = img.crop(box)
                    # Save the tile with a new filename
                    tile_filename = f"{os.path.splitext(filename)[0]}_{i}_{j}.jpg"
                    tile_path = os.path.join(output_folder, tile_filename)
                    tile.save(tile_path)

# Example usage:
input_folder = "E:\dataset\lung classification\正常"
output_folder = "E:\dataset\lung classification\正常"
split_large_images(input_folder, output_folder)
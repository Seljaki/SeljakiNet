import os
import random
import pathlib
from PIL import Image

def rescale_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        try:
            # Open the image
            with Image.open(input_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                # Determine the aspect ratio and resize
                width, height = img.size
                if width < height:
                    new_width = 64
                    new_height = int(64 * height / width)
                else:
                    new_height = 64
                    new_width = int(64 * width / height)

                img = img.resize((new_width, new_height))

                # Randomly crop to 64x64
                left = random.randint(0, new_width - 64)
                top = random.randint(0, new_height - 64)
                right = left + 64
                bottom = top + 64

                img = img.crop((left, top, right, bottom))

                # Save the rescaled image
                output_path = os.path.join(output_dir, filename)
                img.save(output_path)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

def process_images():
    input_path = pathlib.Path("PetImages/Cat")
    input_path.glob("*.jpg")
    output_path = pathlib.Path("processed/cat")
    rescale_images(input_path, output_path)

    input_path = pathlib.Path("PetImages/Dog")
    input_path.glob("*.jpg")
    output_path = pathlib.Path("processed/dog")
    rescale_images(input_path, output_path)

def select_and_move_files(input_dir, output_dir, percentage: float):
    # percentage (float): Percentage of files to move (0-100).
    os.makedirs(output_dir, exist_ok=True)

    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    num_files_to_move = max(1, int(len(all_files) * (percentage / 100)))

    files_to_move = random.sample(all_files, num_files_to_move)

    for filename in files_to_move:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        os.rename(src_path, dst_path)

    print(f"Moved {len(files_to_move)} files to {output_dir}.")

def move_10_percent():
    input_path = pathlib.Path("train_data/clear")
    output_path = pathlib.Path("val_data/clear")
    select_and_move_files(input_path, output_path, 10.0)

    input_path = pathlib.Path("train_data/cloudy")
    output_path = pathlib.Path("val_data/cloudy")
    select_and_move_files(input_path, output_path, 10.0)

    input_path = pathlib.Path("train_data/rainy")
    output_path = pathlib.Path("val_data/rainy")
    select_and_move_files(input_path, output_path, 10.0)

move_10_percent()
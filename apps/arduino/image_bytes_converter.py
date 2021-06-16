import argparse
import json
from PIL import Image

# We assume bytes_path has an alpha channel
CHANNELS = 3
SIZE = 28
def bytes_to_image(bytes_path):
    with open(bytes_path) as f:
        str_bytes = json.load(f)
    list_bytes = [int(s, 16) for s in str_bytes]

    image_out = Image.new("RGB", (SIZE, SIZE))

    for i in range(0, len(list_bytes), CHANNELS):
        pixel = tuple(list_bytes[i:i+3])
        row = (i // CHANNELS) % SIZE
        column = i // (SIZE * CHANNELS)
        image_out.putpixel((row, column), pixel)

    image_out.show()

PROGRAM = """#include <stdint.h>\n
static const uint8_t {}[3072] = {{\n
  {}\n
}};"""

def image_to_c(image_path, out_path, name):
    with Image.open(image_path) as image:
        assert image.size == (32, 32)
        data = []
        for column in range(0, 32):
            for row in range(0, 32):
                byte_tuple = image.getpixel((row, column))
                data.extend(list(byte_tuple))

    str_bytes = list(map(hex, data))
    str_data = ",\n  ".join(str_bytes)
    output = PROGRAM.format(name, str_data)

    with open(out_path, "w") as f:
        f.write(output)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out', type=str, help='output directory for result')
    parser.add_argument('--name', type=str, help='output directory for result')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='file path for input 32x32 image')
    group.add_argument('--bytes', type=str, help='file path for input bytes')
    args = parser.parse_args()

    if args.bytes:
        bytes_to_image(args.bytes)
    elif args.image:
        print(args.image)
        image_to_c(args.image, args.out, args.name)

if __name__ == '__main__':
    main()
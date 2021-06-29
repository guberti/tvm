import argparse


PROGRAM = """#include <stdint.h>\n
static const char {}[3072] = {{
  0x{}
}};"""


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, help='input binary file')
    parser.add_argument('output', type=str, help='output c file')
    parser.add_argument('--name', type=str, default='data', help='variable name')
    args = parser.parse_args()

    with open(args.input, "rb") as in_file:
        input_bytes = in_file.read()

    # Delimiter input to .hex() can only be one character,
    # so we use .replace() as a workaround
    input_hex = input_bytes.hex("-")
    input_hex = input_hex.replace("-", ",\n  0x")

    output_str = PROGRAM.format(args.name, input_hex)
    with open(args.output, "w") as out_file:
        out_file.write(output_str)

if __name__ == '__main__':
    main()

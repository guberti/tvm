import argparse
import pillow

def main():
    parser = argparse.ArgumentParser(description='')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, description='file path for input 32x32 image')
    group.add_argument('--bytes', type=str)

if __name__ == '__main__':
    main()
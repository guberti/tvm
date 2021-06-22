import tensorflow as tf

def image_to_bytes(image_path):
    data = []
    with Image.open(image_path) as image:
        assert image.size == (32, 32)
        for column in range(0, 32):
            for row in range(0, 32):
                byte_tuple = image.getpixel((row, column))
                data.extend(list(byte_tuple))

    return data

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image', type=str, help='file path for input 32x32 image')
    args = parser.parse_args()
    data_list = image_to_bytes(args.image)

    # Load model
    tflite_model_file = "pretrainedResnet_quant.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

if __name__ == '__main__':
    main()

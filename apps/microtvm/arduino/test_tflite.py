import argparse
import json
from collections import Counter

import tvm
from tvm import relay, micro
import tflite

from PIL import Image
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_executor

def load_image(args):
    # Resize it to 224x224
    resized_image = Image.open(args.image).convert('RGB').resize((128, 128))
    img_data = np.asarray(resized_image).astype("uint8")
    img_data = np.expand_dims(img_data, axis=0)
    return img_data


def load_model(args):
    with open(args.input, "rb") as tflite_model_file:
        tflite_model_buf = tflite_model_file.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": (1, 128, 128, 3)},
        dtype_dict={"input": "int8"},
    )

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm", params=params)

    dev = tvm.device("llvm", 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module

def execute(args, input_data, module):
    module.set_input("input", input_data)
    module.run()
    # Why 1001 labels? The first is "I don't know"
    output_shape = (1, 1001)
    tvm_output = module.get_output(0).numpy()
    predictions = tvm_output[0]
    # TODO renormalize
    return predictions


LABELS = "imagenet_labels.json"
def label_predictions(predictions):
    with open(LABELS) as f:
        labels = json.load(f)
    print(predictions)
    print(labels)
    labeled = {labels[i]: predictions[i] for i in range(1001)}
    top = dict(Counter(labeled).most_common(5))
    print("Top five predictions:")
    print(top)
    #for i in range(5):
    #    print("{}: {}".format(labels[top_indices[i]], predictions[top_indices[i]]))

def main():
    # Unused if --mlf-path is specified

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, help='input tflite file')
    parser.add_argument('image', type=str, help='input image')
    args = parser.parse_args()

    input_data = load_image(args)
    module = load_model(args)
    predictions = execute(args, input_data, module)
    label_predictions(predictions)

if __name__ == '__main__':
    main()

import os
import tarfile
import tempfile

import tvm
from tvm import relay, micro
import tflite

# The nrf5340dk target with link-params=1
TARGET = "c -keys=cpu -link-params=1 -mcpu=cortex-m33 -model=nrf5340dk -runtime=c -system-lib=1"
MLF_FILE_TEMPLATE = "{}/output.tar"
def compile_tflite_to_mlf(args):
    with open(args.input, "rb") as tflite_model_file:
        tflite_model_buf = tflite_model_file.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    # TODO update to be non-argument specific
    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": (1, 32, 32, 3)},
        dtype_dict={"input": "int8"},
    )

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, TARGET, params=params)
    tvm.micro.export_model_library_format(mod, MLF_FILE_TEMPLATE.format(args.builddir))


def disassemble_mlf(args):
    with tarfile.open('test.tar', 'r:') as tar:
        my_tar.extractall(args.builddir)


def main():
    # Unused if --mlf-path is specified
    temp_build_dir = tempfile.TemporaryDirectory()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, help='input tflite file')
    parser.add_argument('output', type=str, help='output Arduino project')
    parser.add_argument('--builddir', type=str, help='path to temporary build dir', default=temp_build_dir.name)
    parser.add_argument('--crtdir', type=str, help='path to temporary build dir', default=temp_build_dir.name)
    args = parser.parse_args()

    compile_tflite_to_mlf(args)

    temp_build_dir.cleanup()



if __name__ == '__main__':
    main()

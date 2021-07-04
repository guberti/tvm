import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
from string import Template

import tvm
from tvm import relay, micro
import tflite

# Unlike most C projects, the Arduino IDE will ONLY compile
# code in the top level folder or the src/ subfolder. See
# https://arduino.github.io/arduino-cli/latest/sketch-specification

# Deleting the directory and remaking it screws with the Arduino IDE,
# so we instead just delete the contents of the directory
def prepare_directory(dir_path):
    # Make sure build directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Make sure build directory is empty
    if os.listdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else : # Remove directories
                shutil.rmtree(file_path)


# The nrf5340dk target with link-params=1
TARGET = "c -keys=cpu -link-params=1 -mcpu=cortex-m33 -model=nrf5340dk -runtime=c -system-lib=1"
MLF_FILE_TEMPLATE = "{}/output.tar"
def compile_tflite_to_mlf(args):
    prepare_directory(args.builddir)

    with open(args.input, "rb") as tflite_model_file:
        tflite_model_buf = tflite_model_file.read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    # TODO update to be non-argument specific
    # Ask Josh about the TFLite importer
    mod, params = relay.frontend.from_tflite(tflite_model)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, TARGET, params=params)

    tarpath = MLF_FILE_TEMPLATE.format(args.builddir)
    tvm.micro.export_model_library_format(mod, tarpath)
    return tarpath

def disassemble_mlf(args, tarpath):
    with tarfile.open(tarpath, 'r:') as tar:
        tar.extractall(args.builddir)


def _print_c_array(l):
    c_arr_str = str(l)
    return "{" + c_arr_str[1:-1] + "}"


def _print_c_str(s):
    return '"{}"'.format(s)


DL_DATA_TYPE_REFERENCE = {
    "uint8": "{kDLUInt, 8, 0}",
    "uint16": "{kDLUInt, 16, 0}",
    "uint32": "{kDLUInt, 32, 0}",
    "uint64": "{kDLUInt, 64, 0}",
    "int8": "{kDLInt, 8, 0}",
    "int16": "{kDLInt, 16, 0}",
    "int32": "{kDLInt, 32, 0}",
    "int64": "{kDLInt, 64, 0}",
    "float16": "{kDLFloat, 16, 0}",
    "float32": "{kDLFloat, 32, 0}",
    "float64": "{kDLFloat, 64, 0}",
}


def disassemble_graph_json(obj):
    graph_types = obj["attrs"]["dltype"]
    graph_shapes = obj["attrs"]["shape"]
    assert(graph_types[0] == "list_str")
    assert(graph_shapes[0] == "list_shape")

    return {
        "input_data_dimension": len(graph_shapes[1][0]),
        "input_data_shape": _print_c_array(graph_shapes[1][0]),
        "input_data_type": DL_DATA_TYPE_REFERENCE[graph_types[1][0]],
        "output_data_dimension": len(graph_shapes[1][-1]),
        "output_data_shape": _print_c_array(graph_shapes[1][-1]),
        "output_data_type": DL_DATA_TYPE_REFERENCE[graph_types[1][-1]],
        "input_layer_name": _print_c_str(obj["nodes"][0]["name"]),
    }


GRAPH_JSON_TEMPLATE = 'static const char* graph_json = "{}";\n'
def compile_graph_json(args, obj):
    graph_json = json.dumps(obj).replace('"', '\\"')
    output = GRAPH_JSON_TEMPLATE.format(graph_json)
    graph_json_path = os.path.join(args.output, "src/model/graph_json.c")
    with open(graph_json_path, "w") as out_file:
        out_file.write(output)


SKETCH_NAME_REGEX = r'.*/(.*)'
def _get_sketch_name(args):
    return re.search(SKETCH_NAME_REGEX, args.output).groups()[0]


MLF_DEST_PATH = os.path.join("src", "model")
INO_FILE_PATH = "standalone_template.ino"
def copy_and_populate_template(args):
    # Remove existing folder, if present
    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    # Copy the template dir
    shutil.copytree(args.templatedir, args.output)

    # Rename the .ino file to fit Arduino's convention
    sketch_name = _get_sketch_name(args)
    os.rename(
        os.path.join(args.output, INO_FILE_PATH),
        os.path.join(args.output, sketch_name + ".ino")
    )

    # Copy compiled files from extracted MLF
    # TODO don't copy graph.json, just add graph_json.c
    for source, dest in [
            ("runtime-config/graph/graph.json", "graph.json"),
            ("codegen/host/src/lib0.c", "lib0.c"),
            ("codegen/host/src/lib1.c", "lib1.c"),
            ]:
        shutil.copy(
            os.path.join(args.builddir, source),
            os.path.join(args.output, MLF_DEST_PATH, dest)
        )

    # Load graph.json, serialize to c format, and extact parameters
    with open(os.path.join(args.output, MLF_DEST_PATH, "graph.json")) as f:
        graph_data = json.load(f)
    compile_graph_json(args, graph_data)
    template_values = disassemble_graph_json(graph_data)

    # Use extracted parameters to produce parameters.h
    with open(args.templateparams, 'r') as f:
        template_params = Template(f.read())
    parameters_h = template_params.substitute(template_values)
    with open(os.path.join(args.output, "src/parameters.h"), "w") as out_file:
        out_file.write(parameters_h)


IMPORT_REGEX = r'^#include[\s]+(?:<|")([0-9A-Za-z/\-_]+\.(?:h|cpp|c))(?:>|")'
# TODO make this fix all imports, not just these few
IMPORT_PATTERNS = {
    "tvm/runtime/crt/module.h": '"../standalone_crt/include/tvm/runtime/crt/module.h"',
    "tvm/runtime/c_runtime_api.h": '"../standalone_crt/include/tvm/runtime/c_runtime_api.h"',
    "tvm/runtime/c_backend_api.h": '"../standalone_crt/include/tvm/runtime/c_backend_api.h"',
}

def redirect_imports(args, input_file_path):
    file_path = os.path.join(args.output, input_file_path)
    with open(file_path, 'r') as f:
        c_file_lines = f.readlines()

    check_is_import = re.compile(IMPORT_REGEX)
    for i in range(len(c_file_lines)):
        result = check_is_import.search(c_file_lines[i])
        if result:
            import_path = result.group(1)
            if import_path in IMPORT_PATTERNS:
                c_file_lines[i] = re.sub(
                    r'("|<)' + import_path + r'("|>)',
                    IMPORT_PATTERNS[import_path],
                    c_file_lines[i],
                )

    with open(file_path, 'w') as f:
        f.writelines(c_file_lines)


TEMPLATE_DIR = "standalone_template"
TEMPLATE_PARAMS = "parameters_template.h"
def main():
    # Unused if --mlf-path is specified
    temp_build_dir = tempfile.TemporaryDirectory()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, help='input tflite file')
    parser.add_argument('output', type=str, help='output Arduino project')
    parser.add_argument('--builddir', type=str, help='path to temporary build dir', default=temp_build_dir.name)
    parser.add_argument('--templatedir', type=str, help='path to template dir', default=TEMPLATE_DIR)
    parser.add_argument('--templateparams', type=str, help='path to template params', default=TEMPLATE_PARAMS)
    args = parser.parse_args()

    tarpath = compile_tflite_to_mlf(args)
    disassemble_mlf(args, tarpath)
    copy_and_populate_template(args)

    # TODO make this better
    redirect_imports(args, "src/model/lib0.c")
    redirect_imports(args, "src/model/lib1.c")
    temp_build_dir.cleanup()



if __name__ == '__main__':
    main()

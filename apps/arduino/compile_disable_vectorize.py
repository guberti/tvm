import os
import tvm
from tvm import relay, micro
import tflite

tflite_model_file = "pretrainedResnet_quant.tflite"
tflite_model_buf = open(tflite_model_file, "rb").read()
tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={"input": (1, 32, 32, 3)}, dtype_dict={"input": "int8"}
)

target = "c"
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    mod = relay.build(mod, target, params=params)
tvm.micro.export_model_library_format(mod, "output.tar")


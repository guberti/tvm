static const char* graph_json = "{\"nodes\": [{\"op\": \"null\", \"name\": \"Reshape_1\", \"inputs\": []}, {\"op\": \"null\", \"name\": \"p0\", \"inputs\": []}, {\"op\": \"tvm_op\", \"name\": \"fused_reshape_cast_subtract_1\", \"attrs\": {\"num_outputs\": \"1\", \"num_inputs\": \"2\", \"flatten_data\": \"0\", \"func_name\": \"fused_reshape_cast_subtract_1\", \"hash\": \"9d4bf10a2292c188\"}, \"inputs\": [[0, 0, 0], [1, 0, 0]]}, {\"op\": \"null\", \"name\": \"p1\", \"inputs\": []}, {\"op\": \"null\", \"name\": \"p2\", \"inputs\": []}, {\"op\": \"null\", \"name\": \"p3\", \"inputs\": []}, {\"op\": \"null\", \"name\": \"p4\", \"inputs\": []}, {\"op\": \"null\", \"name\": \"p5\", \"inputs\": []}, {\"op\": \"tvm_op\", \"name\": \"fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip\", \"attrs\": {\"num_outputs\": \"1\", \"num_inputs\": \"6\", \"flatten_data\": \"0\", \"func_name\": \"fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip\", \"out_layout\": \"\", \"kernel_layout\": \"HWIO\", \"data_layout\": \"NHWC\", \"hash\": \"27ee460434e6faee\"}, \"inputs\": [[2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0]]}, {\"op\": \"null\", \"name\": \"p6\", \"inputs\": []}, {\"op\": \"tvm_op\", \"name\": \"fused_reshape_cast_subtract\", \"attrs\": {\"num_outputs\": \"1\", \"num_inputs\": \"2\", \"flatten_data\": \"0\", \"func_name\": \"fused_reshape_cast_subtract\", \"hash\": \"176bd09511aee12b\"}, \"inputs\": [[8, 0, 0], [9, 0, 0]]}, {\"op\": \"null\", \"name\": \"p7\", \"inputs\": []}, {\"op\": \"null\", \"name\": \"p8\", \"inputs\": []}, {\"op\": \"tvm_op\", \"name\": \"fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_\", \"attrs\": {\"num_outputs\": \"1\", \"num_inputs\": \"3\", \"flatten_data\": \"0\", \"func_name\": \"fused_nn_contrib_dense_pack_add_fixed_point_multiply_add_clip_cast_cast_subtract_14669711146056581479_\", \"hash\": \"a78a02f28f4677aa\"}, \"inputs\": [[10, 0, 0], [11, 0, 0], [12, 0, 0]]}, {\"op\": \"tvm_op\", \"name\": \"fused_nn_softmax\", \"attrs\": {\"num_outputs\": \"1\", \"num_inputs\": \"1\", \"flatten_data\": \"0\", \"func_name\": \"fused_nn_softmax\", \"hash\": \"e525638339182d2d\"}, \"inputs\": [[13, 0, 0]]}, {\"op\": \"tvm_op\", \"name\": \"fused_divide_add_round_cast_clip_cast\", \"attrs\": {\"num_outputs\": \"1\", \"num_inputs\": \"1\", \"flatten_data\": \"0\", \"func_name\": \"fused_divide_add_round_cast_clip_cast\", \"hash\": \"1f81d9f43de9b085\"}, \"inputs\": [[14, 0, 0]]}], \"arg_nodes\": [0, 1, 3, 4, 5, 6, 7, 9, 11, 12], \"heads\": [[15, 0, 0]], \"attrs\": {\"dltype\": [\"list_str\", [\"int8\", \"int16\", \"int16\", \"int16\", \"int32\", \"int64\", \"int64\", \"int64\", \"int8\", \"int16\", \"int16\", \"int16\", \"int32\", \"float32\", \"float32\", \"int8\"]], \"storage_id\": [\"list_int\", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 10, 11, 12, 13, 12]], \"shape\": [\"list_shape\", [[1, 1960], [], [1, 49, 40, 1], [10, 8, 1, 8], [1, 1, 1, 8], [1, 1, 1, 8], [1, 1, 1, 8], [1, 1, 1, 8], [1, 25, 20, 8], [], [1, 4000], [1, 4000, 4], [4], [1, 4], [1, 4], [1, 4]]]}, \"node_row_ptr\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}";

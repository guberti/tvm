#ifndef Parameters_h
#define Parameters_h

#include "src/standalone_crt/include/dlpack/dlpack.h"

// Some Arduinos (like the Spresense) have multiple CPUs,
// so this could be expaneded at some point
static const DLDevice HARDWARE_DEVICE = {kDLCPU, 0};

static const int INPUT_DATA_DIMENSION = 4;
static const int64_t INPUT_DATA_SHAPE[] = {1, 32, 32, 3};
static const DLDataType INPUT_DATA_TYPE = {kDLInt, 8, 0};
static const char* INPUT_LAYER = "input_1_int8";

static const int OUTPUT_DATA_DIMENSION = 2;
static const int64_t OUTPUT_DATA_SHAPE[] = {1, 10};
static const DLDataType OUTPUT_DATA_TYPE = {kDLInt, 8, 0};

#endif

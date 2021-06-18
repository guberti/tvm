#include "Tvmq.h"
#include <Camera.h>
#include <SDHCI.h> 
#include <stdint.h>

//// Labels for naming images ////
static char LABELS[10][10] = {
  "CAR",
  "PLANE",
  "BIRD",
  "CAT",
  "DEER",
  "DOG",
  "FROG",
  "HORSE",
  "SHIP",
  "TRUCK",
};

//// Global variables ////
SDClass theSD;
static uint8_t INPUT_BUF[3072];
static Tvmq t;
int take_picture_count = 0;

//// Function called by camera streaming ////
// We MUST use camera streaming, as static images
// can only be output as JPEG
void CamCB(CamImage img) {
  Serial.println("Took picture"); 
  Serial.flush();

  // We cannot use the built-in image crop/resize,
  // as it only works for YUV422 images (note that
  // this behavior is undocumented). We also cannot
  // change the image format afterwards. Thus, we
  // do it ourselves.
  
  //// Perform image resize ////
  img.convertPixFormat(CAM_IMAGE_PIX_FMT_RGB565);
  uint16_t* originalBuf = (uint16_t*)img.getImgBuff();
  uint16_t outIndex = 0;
  
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 32; ++j) {
      uint16_t red = 0;
      uint16_t green = 0;
      uint16_t blue = 0;
      
      for (int m = 0; m < 7; m++) {
        for (int n = 0; n < 7; n++) {
          uint32_t sp_row = 8 + 7 * i + m;
          uint32_t sp_col = 48 + 7 * j + n;
          uint32_t index = 320 * sp_row + sp_col;
          
          uint16_t pixel = originalBuf[index];
          red += (pixel & 0xF800) >> 8;
          green += (pixel & 0x07E0) >> 3;
          blue += (pixel & 0x001F) << 3;
        }
      }

      INPUT_BUF[outIndex] = (uint8_t) (red / 49);
      INPUT_BUF[outIndex + 1] = (uint8_t) (green / 49);
      INPUT_BUF[outIndex + 2] = (uint8_t) (blue / 49);
      outIndex += 3;
    }
  }

  //// Perform inference ////
  int category = t.infer_category(INPUT_BUF);
  
  Serial.print("Identified image as: ");
  Serial.println(LABELS[category]);
  
  char filename[32] = {0};
  sprintf(filename, "IF2_%s_%03d.rgb8", LABELS[category], take_picture_count);
  theSD.remove(filename);
  File largeFile = theSD.open(filename, FILE_WRITE);
  largeFile.write(INPUT_BUF, 3072);
  largeFile.close(); 
  
  Serial.println("Finished writing file"); 
  take_picture_count++;
}

void setup() {
  Serial.begin(9600);
  t = Tvmq();
  theCamera.begin();
  theCamera.startStreaming(true, CamCB);
}

void loop() {}

#include<cstdlib>
#include<cstdio>
#include<string>
#include<cmath>
#include<iostream>
#include "ff.h"
#include "xparameters.h"
#include "xdevcfg.h"

using namespace std;

#define SCALE 512


static FATFS fatfs;

int SD_Init();

int read_int16(string Filename, int16_t * x, int length);
int write_int16(string Filename, int16_t * x, int length);
int read_int8(string Filename, int8_t * x, int length);
int write_int8(string Filename, int8_t * x, int length);
int read_int32(string Filename, int32_t * x, int length);
int write_int32(string Filename, int32_t * x, int length);




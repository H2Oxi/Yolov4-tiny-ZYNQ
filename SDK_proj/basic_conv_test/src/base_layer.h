#include <iostream>
#include <vector>   
#include <cstring>
#include "sd_io.h"
using namespace std;

#define data_t int8_t 

const string syn_data_dir = "hw_tst/conv1_relu0125/int8_m16/";
const string syn_int8m16_dir = "my_int_weights_relu0125/int8_M16/";


int load_int8_data(int8_t *buffer , const string dir,int size);
int load_int16_data(int16_t *buffer , const string dir,int size);
int load_int32_data(int32_t *buffer , const string dir,int size);
int save_int32_data(int32_t *buffer ,const string dir ,int size);
int save_int8_data(int8_t *buffer ,const string dir ,int size);
void print_out_tensor(int32_t * out ,int cho,int width);
void tensor_int32_init(int32_t * tensor,int size );
void tensor_int8_init(int8_t * tensor,int size );
void tensor2int8(int32_t * in,int8_t * out,int size0,int size1,int size2);
void concatenate(int8_t * tensor0,int8_t * tensor1, int8_t * out,int channels,int width);
void concatenate_P5_feat1(int8_t * P5_Upsample,int8_t * feat1, int8_t * out);


data_t*** padArray(const data_t * originalArray, int originalRows, int originalCols,int chi);
data_t * flatten_3d(data_t *** arr,int size2,int size1,int size0);

class conv_2d
{
    private:
        
    public:
        unsigned int in_shape[3]; //in_channels,r,c
        
        unsigned int output_channels;
        unsigned int in_channels;
        unsigned int ksize;
        unsigned int stride;

        data_t * weights;
        int32_t * bias;
        int16_t m0;
        unsigned int out_shape[3];
        void init(unsigned int in_shape[3],unsigned int output_channels,unsigned int ksize,unsigned int stride );
        void data_update(data_t * weights,int32_t * bias,int16_t m0);
        void forward(data_t *x,int32_t *out);
        void padding_forward(data_t *x ,int32_t *out);
      
};

class leakyrelu
{
    private:
        

    public:
        unsigned int in_shape[3];
        unsigned short int S_relu;
        unsigned int S_bitshift;
        void init(unsigned int in_shape_i[3])  ;
        void forward(data_t *x,int32_t *out);
        void S_update(unsigned short int S_relu,unsigned int S_bitshift) ;
        

};

class maxpooling
{
    private:
        
    public:
        unsigned int in_shape[3];
        
        unsigned int ksize;
        unsigned int stride;
        data_t find_max(data_t x0,data_t x1,data_t x2,data_t x3);
        unsigned int out_shape[3];
        void init(unsigned int in_shape[3],unsigned int ksize,unsigned int stride );
        void forward(data_t *x,data_t *out);
};

class basic_conv
{
    private:

    public:
        conv_2d conv;
        leakyrelu activation;
        void init(unsigned int in_shape[3],unsigned int output_channels,unsigned int ksize,unsigned int stride);
        void forward(data_t *x,data_t *out);
        void data_update(const string file_dir);

};

class resblock_body
{
    public:
        unsigned int in_shape[3];
        unsigned int output_channels;
        string name;

        basic_conv conv1;
        basic_conv conv2;
        basic_conv conv3;
        basic_conv conv4;
        maxpooling maxpool1;

        void init(string name ,unsigned int in_shape[3] , unsigned int output_channels );
        void forward(data_t *x,data_t *out0,data_t *out1);

};

class csp_dark_net
{
    public:
        basic_conv conv1;
        basic_conv conv2;
        resblock_body resblock_body1;
        resblock_body resblock_body2;
        resblock_body resblock_body3;
        basic_conv conv3;

        void init();
        void forward(data_t *x,data_t *out0,data_t *out1);
    
};

class upsampling
{
    public:
        unsigned int in_shape[3];
        unsigned int out_shape[3];

        void init(unsigned int in_shape[3]);
        void forward(data_t *x,data_t *out);
};

class yolo_head
{
    public:
        unsigned int in_shape[3];
        unsigned int center_channels;
        unsigned int center_shape[3];
        unsigned int out_channels;
        string name;

        basic_conv conv1;
        conv_2d conv2;

        void init(unsigned int in_shape[3],unsigned int center_channels,unsigned int out_channels ,string name);
        void forward(data_t *x,data_t *out);
};

class up_sample
{
    public:
        unsigned int in_shape[3];

        basic_conv conv1;
        upsampling upsample1;

        void init(unsigned int in_shape[3]);
        void forward(data_t *x,data_t *out);

};

class yolo_body
{
    public:
        csp_dark_net backbone;
        basic_conv conv_for_p5;
        yolo_head yolohead_p5;
        yolo_head yolohead_p4;
        up_sample up_sample0;

        void init();
        void forward(data_t *x,data_t *out0,data_t *out1)      ;
};

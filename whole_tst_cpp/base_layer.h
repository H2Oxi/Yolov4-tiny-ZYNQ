#include <iostream>
#include <vector>   
#include <cstring>
using namespace std;

#define data_t int8_t 
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
#include <iostream>  
#include <fstream>  
#include <vector>  
#include <string>  

#include "base_layer.h"

using namespace std; 

//void load_int_data(int8_t *buffer , dir);


//////////////////
void conv_2d::init(unsigned int in_shape_i[3],unsigned int output_channels_i,unsigned int ksize_i,unsigned int stride_i )
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape_i[i];
    }
    this->output_channels=output_channels_i;
    this->ksize=ksize_i;
    this->stride=stride_i;
    this->in_channels=this->in_shape[0];
    this->out_shape[0]=this->output_channels;
    this->out_shape[1]=this->in_shape[1]/this->stride;
    this->out_shape[2]=this->in_shape[2]/this->stride;

}

void conv_2d::data_update(data_t * weights,int32_t * bias , int16_t m0)
{
    this->weights=weights;
    this->bias = bias;
    this-> m0=m0;

}


void conv_2d::forward(data_t *x ,int32_t *out)
{


	//Kernel_Row:
	for(int kr=0;kr<ksize;kr++)
	{
		//Kernel_Col:
		for(int kc=0;kc<ksize;kc++)
		{
			//Row:
			for(int r=0;r<out_shape[1];r++)
			{
				//Column:
				for(int c=0;c<out_shape[2];c++)
				{
					//Out_channel:
					for(int o=0;o<output_channels;o++)
					{
                        int out_offset= (o*out_shape[1]+r)*out_shape[2]+c  ;//((out_shape[1]*c)+r)*out_shape[2]+o;
						//in_channel:
						for(int in=0;in<in_channels;in++)
						{
                            //out[o][r][c]+=x[in][stride*r+kr][stride*c+kc]*this->weights[o][in][kr][kc]+this->bias[o]
                            int x_offset=(in*(in_shape[1])+stride*r+kr)*(in_shape[2])+stride*c+kc;//((stride*c+kc)*(this->in_shape[2])+stride*r+kr)*(this->in_shape[1])+in;
                            int w_offset=((o*in_channels+in)*ksize+kr)*ksize+kc;//((kc*ksize+kr)*ksize+in)*in_channels+o;
                     
                            *(out+out_offset)+= (* (x +x_offset))* (*(weights+w_offset));               
                            
                         
						}
                        if((kc==ksize-1)&&(kr==ksize-1))
                        {
                            *(out+out_offset) += bias[o];
                            *(out+out_offset) = ((*(out+out_offset)) * this->m0 )>>15;

                        }
						
					}

				}
			}
		}
	}

}

void conv_2d::padding_forward(data_t *x ,int32_t *out)
{


	//Kernel_Row:
	for(int kr=0;kr<ksize;kr++)
	{
		//Kernel_Col:
		for(int kc=0;kc<ksize;kc++)
		{
			//Row:
			for(int r=0;r<out_shape[1];r++)
			{
				//Column:
				for(int c=0;c<out_shape[2];c++)
				{
					//Out_channel:
					for(int o=0;o<output_channels;o++)
					{
                        int out_offset= (o*out_shape[1]+r)*out_shape[2]+c  ;//((out_shape[1]*c)+r)*out_shape[2]+o;
						//in_channel:
						for(int in=0;in<in_channels;in++)
						{
                            
                            if( ((stride*r+kr)==0)||((stride*c+kc)==0)||( (stride*c+kc)==1+in_shape[2] )|| ( (stride*r+kr)==1+in_shape[1] ))
                            {

                            }
                            else{
                            int x_offset=(in*(in_shape[1])+stride*r+kr-1)*(in_shape[1])+stride*c+kc-1;//((stride*c+kc)*(this->in_shape[2])+stride*r+kr)*(this->in_shape[1])+in;
                            int w_offset=((o*in_channels+in)*ksize+kr)*ksize+kc; //cho,chi,kr,kc       
                            
                     
                            *(out+out_offset)+= (* (x +x_offset))* (*(weights+w_offset));
                            /*if(out_offset==0)
                            {
                                std::cout<<"x "<< in << " "<< (stride*r+kr-1)<<" "<< (stride*c+kc-1) << "="<< static_cast<int>(* (x +x_offset))<<"(x_offset:"<<x_offset<<")";
                                std::cout<<"w "<< o << " "<< in << " "<< kr << " "<< kc << "="<<static_cast<int>(*(weights+w_offset))<<"(w_offset:"<<w_offset<<")";
                                std::cout<<"out[0]="<<*(out+0);
                                std::cout<<endl;
                            }
                            */
                            }

                            
                            
						}
                        if((kc==ksize-1)&&(kr==ksize-1))
                        {
                            *(out+out_offset) += bias[o];
                            /*if(out_offset==0)
                                std::cout<<"out["<< out_offset <<"]="<<*(out+out_offset);*/
                            *(out+out_offset) = ((*(out+out_offset)) * this->m0 )>>15;
                            /*if(out_offset==0)
                            {
                                std::cout<<"out["<< out_offset <<"]="<<*(out+out_offset);
                                std::cout<<endl;
                            }*/
                        }
						
					}

				}
			}
		}
	}

}

//////////////////////////////
void leakyrelu::init(unsigned int in_shape_i[3]) 
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape_i[i];
    }
    

}
void leakyrelu::S_update(unsigned short int S_relu,unsigned int S_bitshift) 
{
    this->S_relu=S_relu;
    this->S_bitshift=S_bitshift;
}



void leakyrelu::forward(data_t *x,int32_t *out)
{
	//Row:
	for(int r=0;r<in_shape[1];r++)
	{
		//Column:
		for(int c=0;c<in_shape[2];c++)
		{
		    //Out_channel:
			for(int o=0;o<in_shape[0];o++)
			{
                int x_offset=(o*in_shape[1]+r)*in_shape[2]+c;
                if(* (x +x_offset)>=0)
                {
                    *(out+x_offset) = (* (x +x_offset)) * S_relu ;
                    *(out+x_offset) = *(out+x_offset) >> (S_bitshift-1);
                }
                else
                {
                    *(out+x_offset) = (* (x +x_offset)) * S_relu ;
                    *(out+x_offset) = *(out+x_offset) >> (S_bitshift-1+3);
                }

			}
		}
	}
}





void maxpooling::init(unsigned int in_shape[3],unsigned int ksize,unsigned int stride )
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape[i];
    }
    this->out_shape[0]=this->in_shape[0];
    this->ksize=ksize;
    this->stride=stride;
    this->out_shape[1]=this->in_shape[1]/this->stride;
    this->out_shape[2]=this->in_shape[2]/this->stride;

}

void maxpooling::forward(data_t *x,data_t *out)
{
	//Row:
	for(int r=0;r<out_shape[1];r++)
	{
		//Column:
		for(int c=0;c<out_shape[2];c++)
		{
		    //Out_channel:
			for(int o=0;o<out_shape[0];o++)
			{
                int x_offset0=(o*in_shape[1]+2*r)*in_shape[2]+2*c;
                int x_offset1=(o*in_shape[1]+2*r)*in_shape[2]+2*c+1;
                int x_offset2=(o*in_shape[1]+2*r+1)*in_shape[2]+2*c;
                int x_offset3=(o*in_shape[1]+2*r+1)*in_shape[2]+2*c+1;

                int out_offset=(o*out_shape[1]+r)*out_shape[2]+c;
              
                *(out+out_offset) = maxpooling::find_max(*(x+x_offset0) ,*(x+x_offset1),*(x+x_offset2),*(x+x_offset3)) ;
			}
		}
	}
}

data_t maxpooling::find_max(data_t x0,data_t x1,data_t x2,data_t x3)
{
    data_t max=x0;
    if(max<x1)
        max=x1;
    else
        max=max;
    if(max<x2)
        max=x2;
    else
        max=max;
    if(max<x3)
        max=x3;
    else
        max=max;  
    return max;  
}
////////////////////////////////

void basic_conv::init(unsigned int in_shape[3],unsigned int output_channels,unsigned int ksize,unsigned int stride)
{
    this->conv.init(in_shape ,output_channels ,ksize ,stride);
    this->activation.init(this->conv.out_shape);
    
}

void basic_conv::data_update(const string file_dir)
{
    //int32_t *bias;
    conv.bias = new int32_t[this->conv.output_channels];
    //int8_t *weights;

    conv.weights = new int8_t[conv.output_channels*conv.in_channels*conv.ksize*conv.ksize]; 
    
    
    int16_t m0 [1];
    int16_t s0 [1];

    load_int8_data(conv.weights ,  file_dir +"w_q.bin" , conv.output_channels*conv.in_channels*conv.ksize*conv.ksize); 
    load_int16_data(m0 , file_dir +"m_0.bin" , 1);

    cout<< * m0 <<endl;
    
    load_int32_data(conv.bias ,  file_dir +"b_q.bin" ,  conv.output_channels);



    conv.data_update(&conv.weights[0],&conv.bias[0],* m0);

    load_int16_data(s0 , file_dir +"S_relu_q.bin" , 1);
    activation.S_update(* s0,15);

}

void basic_conv::forward(data_t *x,data_t *out)
{
    int32_t *buffer1;
    buffer1 = new int32_t[conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]];
    fill_n(buffer1,conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2],0);
    //tensor_int32_init(buffer1,conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]);
    if(conv.ksize==3)
        conv.padding_forward(x,buffer1);
    else
        conv.forward(x,buffer1);
    delete [] conv.bias;
    delete [] conv.weights;
    

    int8_t *conv_out;
    conv_out = new int8_t[conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]];

    tensor2int8(buffer1,conv_out,conv.out_shape[0],conv.out_shape[1],conv.out_shape[2]);
    
    

    delete [] buffer1;

    int32_t *buffer2;
    buffer2 = new int32_t[conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]];
    
    tensor_int32_init(buffer2,conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]);

    activation.forward(conv_out,buffer2);
    delete [] conv_out;

    tensor2int8(buffer2,out,conv.out_shape[0],conv.out_shape[1],conv.out_shape[2]);
    delete [] buffer2;

}

////////////////////////////////

void resblock_body::init(string name ,unsigned int in_shape[3] , unsigned int output_channels )
{
    this->name=name;
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape[i];
    }
    this->output_channels = output_channels;
    this->conv1.init(this->in_shape,output_channels,3,1);
    this->conv4.init(this->in_shape,output_channels,1,1);
    this->in_shape[0]=in_shape[0]/2;
    this->conv2.init(this->in_shape,output_channels/2,3,1);
    this->conv3.init(this->in_shape,output_channels/2,3,1);

    this->in_shape[0]=in_shape[0]*2;
    this->maxpool1.init(this->in_shape,2,2);

}

void resblock_body::forward(data_t *x,data_t *out0,data_t *out1)
{
    string data_dir=syn_int8m16_dir + "backbone/";
    //conv1
    conv1.data_update(data_dir+name+"/conv1/");
    int8_t *Out1;
    Out1 = new int8_t[conv1.conv.in_shape[0]*conv1.conv.in_shape[1]*conv1.conv.in_shape[2]];
    fill_n(Out1,conv1.conv.in_shape[0]*conv1.conv.in_shape[1]*conv1.conv.in_shape[2],0);
    conv1.forward(x,Out1);
    delete [] x;
    //conv2
    conv2.data_update(data_dir+name+"/conv2/");
    int8_t *Out2;
    Out2 = new int8_t[conv2.conv.in_shape[0]*conv2.conv.in_shape[1]*conv2.conv.in_shape[2]];
    fill_n(Out2,conv2.conv.in_shape[0]*conv2.conv.in_shape[1]*conv2.conv.in_shape[2],0);
    conv2.forward(&Out1[conv1.conv.in_shape[0]*conv1.conv.in_shape[1]*conv1.conv.in_shape[2]/2-1],Out2);
    //conv3
    conv3.data_update(data_dir+name+"/conv3/");
    int8_t *Out3;
    Out3 = new int8_t[conv3.conv.in_shape[0]*conv3.conv.in_shape[1]*conv3.conv.in_shape[2]];
    fill_n(Out3,conv3.conv.in_shape[0]*conv3.conv.in_shape[1]*conv3.conv.in_shape[2],0);
    conv3.forward(Out2,Out3);
    //conv4
    conv4.data_update(data_dir+name+"/conv4/");
    int8_t *in4;
    in4 = new int8_t[conv4.conv.in_shape[0]*conv4.conv.in_shape[1]*conv4.conv.in_shape[2]];
    fill_n(in4,conv4.conv.in_shape[0]*conv4.conv.in_shape[1]*conv4.conv.in_shape[2],0);
    //int8_t *Out4;//feat
    //Out4 = new int8_t[conv4.conv.in_shape[0]*conv4.conv.in_shape[1]*conv4.conv.in_shape[2]];
    //fill_n(Out4,conv4.conv.in_shape[0]*conv4.conv.in_shape[1]*conv4.conv.in_shape[2],0);
    /*
    cout<<"out2-size:"<<conv2.conv.in_shape[0]<<"  "<<conv2.conv.in_shape[1]<<endl;
    cout<<"out3-size:"<<conv3.conv.in_shape[0]<<"  "<<conv3.conv.in_shape[1]<<endl;
    cout<<"in4-size:"<<conv4.conv.in_shape[0]<<"  "<<conv4.conv.in_shape[1]<<endl;
    */
    concatenate(Out3,Out2,in4,conv2.conv.in_shape[0],conv2.conv.in_shape[1]);
    delete [] Out2;
    delete [] Out3;
    conv4.forward(in4,out1);
    delete [] in4;
    //maxpool1
    int8_t *in5;
    in5 = new int8_t[maxpool1.in_shape[0]*maxpool1.in_shape[1]*maxpool1.in_shape[2]];
    fill_n(in5,maxpool1.in_shape[0]*maxpool1.in_shape[1]*maxpool1.in_shape[2],0);
    concatenate(Out1,out1,in5,conv1.conv.in_shape[0],conv1.conv.in_shape[1]);
    //int8_t *Out5;
    //Out5 = new int8_t[maxpool1.out_shape[0]*maxpool1.out_shape[1]*maxpool1.out_shape[2]];
    //fill_n(Out5,maxpool1.out_shape[0]*maxpool1.out_shape[1]*maxpool1.out_shape[2],0);
    delete [] Out1;
    maxpool1.forward(in5,out0);
    delete [] in5;
    //
    //out0=Out5;
    //out1=Out4;
}


//////////////////////////////////

void csp_dark_net::init()
{
    unsigned int conv1_in_shape[]={3,416,416};
    this->conv1.init(conv1_in_shape,32,3,2);
    unsigned int conv2_in_shape[]={32,208,208};
    this->conv2.init(conv2_in_shape,64,3,2);
    unsigned int resblock1_in_shape[]={64,104,104};
    this->resblock_body1.init("resblock_body1",resblock1_in_shape,64);
    unsigned int resblock2_in_shape[]={128,52,52};
    this->resblock_body2.init("resblock_body2",resblock2_in_shape,128);
    unsigned int resblock3_in_shape[]={256,26,26};
    this->resblock_body3.init("resblock_body3",resblock3_in_shape,256);
    unsigned int conv3_in_shape[]={512,13,13};
    this->conv3.init(conv3_in_shape,512,3,1);

}

void csp_dark_net::forward(data_t *x,data_t *out0,data_t *out1)
{
    string data_dir=syn_int8m16_dir + "backbone/";
    //conv1
    conv1.data_update(data_dir+"/conv1/"); 
    int8_t *Out1;
    Out1 = new int8_t[conv1.conv.out_shape[0]*conv1.conv.out_shape[1]*conv1.conv.out_shape[2]];
    fill_n(Out1,conv1.conv.out_shape[0]*conv1.conv.out_shape[1]*conv1.conv.out_shape[2],0);
    conv1.forward(x,Out1);   
    delete [] x;
    //conv2
    conv2.data_update(data_dir+"/conv2/"); 
    int8_t *Out2;
    Out2 = new int8_t[conv2.conv.out_shape[0]*conv2.conv.out_shape[1]*conv2.conv.out_shape[2]];
    fill_n(Out2,conv2.conv.out_shape[0]*conv2.conv.out_shape[1]*conv2.conv.out_shape[2],0);
    conv2.forward(Out1,Out2);
    delete [] Out1;
    //resblock_body1
    int8_t *out0_res1;
    int8_t *out1_res1;
    out0_res1 = new int8_t[resblock_body1.maxpool1.out_shape[0]*resblock_body1.maxpool1.out_shape[1]*resblock_body1.maxpool1.out_shape[2]];
    fill_n(out0_res1,resblock_body1.maxpool1.out_shape[0]*resblock_body1.maxpool1.out_shape[1]*resblock_body1.maxpool1.out_shape[2],0);
    out1_res1 = new int8_t[resblock_body1.conv4.conv.in_shape[0]*resblock_body1.conv4.conv.in_shape[1]*resblock_body1.conv4.conv.in_shape[2]];
    fill_n(out1_res1,resblock_body1.conv4.conv.in_shape[0]*resblock_body1.conv4.conv.in_shape[1]*resblock_body1.conv4.conv.in_shape[2],0);

    resblock_body1.forward(Out2,out0_res1,out1_res1);
    
    delete [] out1_res1;
    //resblock_body2
    int8_t *out0_res2;
    int8_t *out1_res2;
    out0_res2 = new int8_t[resblock_body2.maxpool1.out_shape[0]*resblock_body2.maxpool1.out_shape[1]*resblock_body2.maxpool1.out_shape[2]];
    fill_n(out0_res2,resblock_body2.maxpool1.out_shape[0]*resblock_body2.maxpool1.out_shape[1]*resblock_body2.maxpool1.out_shape[2],0);
    out1_res2 = new int8_t[resblock_body2.conv4.conv.in_shape[0]*resblock_body2.conv4.conv.in_shape[1]*resblock_body2.conv4.conv.in_shape[2]];
    fill_n(out1_res2,resblock_body2.conv4.conv.in_shape[0]*resblock_body2.conv4.conv.in_shape[1]*resblock_body2.conv4.conv.in_shape[2],0);
    resblock_body2.forward(out0_res1,out0_res2,out1_res2);
    
    delete [] out1_res2;
    //resblock_body3
    int8_t *out0_res3;
    out0_res3 = new int8_t[resblock_body3.maxpool1.out_shape[0]*resblock_body3.maxpool1.out_shape[1]*resblock_body3.maxpool1.out_shape[2]];
    resblock_body3.forward(out0_res2,out0_res3,out0);
    
    //conv4

    fill_n(out1,conv3.conv.out_shape[0]*conv3.conv.out_shape[1]*conv3.conv.out_shape[2],0);
    conv3.data_update(data_dir+"/conv3/"); 
    conv3.forward(out0_res3,out1);
    delete [] out0_res3;
    //
    //out0=out1_res3;
    //out1=out_conv3;

}


/////////////////////////////////

void upsampling::init(unsigned int in_shape[3])
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape[i];
    }
    this->out_shape[0]=in_shape[0];
    this->out_shape[1]=2*in_shape[1];
    this->out_shape[2]=2*in_shape[2];
}


void upsampling::forward(data_t *x,data_t *out)
{
    //Row:
	for(int r=0;r<in_shape[1];r++)
	{
		//Column:
		for(int c=0;c<in_shape[2];c++)
		{
		    //Out_channel:
			for(int o=0;o<in_shape[0];o++)
			{
                //output[o][r * 2][c * 2] = input[o][r][c]; 
                out[(o*out_shape[1]+2*r)*out_shape[2]+2*c]=x[(o*in_shape[1]+r)*in_shape[2]+c];
                out[(o*out_shape[1]+2*r+1)*out_shape[2]+2*c]=x[(o*in_shape[1]+r)*in_shape[2]+c];
                out[(o*out_shape[1]+2*r)*out_shape[2]+2*c+1]=x[(o*in_shape[1]+r)*in_shape[2]+c];
                out[(o*out_shape[1]+2*r+1)*out_shape[2]+2*c+1]=x[(o*in_shape[1]+r)*in_shape[2]+c];
            }
        }
    }
}

///////////////////////////////

void yolo_head::init(unsigned int in_shape[3],unsigned int center_channels,unsigned int out_channels,string name)
{
    this->center_channels=center_channels;
    this->out_channels=out_channels;
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape[i];
    }    

    conv1.init(this->in_shape,this->center_channels,3,1);

    for(int i=0;i<3;i++)
    {
        this->center_shape[i]=this->conv1.conv.out_shape[i];
    } 
    conv2.init(this->center_shape,this->out_channels,1,1);
    this->name=name;
}

void yolo_head::forward(data_t *x,data_t *out)
{
    string data_dir=syn_int8m16_dir + this->name;
    //conv1
    conv1.data_update(data_dir+"/conv1/"); 
    int8_t *Out1;
    Out1 = new int8_t[conv1.conv.out_shape[0]*conv1.conv.out_shape[1]*conv1.conv.out_shape[2]];
    conv1.forward(x,Out1);
    
    //conv2
    conv2.bias = new int32_t[this->conv2.output_channels];
    conv2.weights = new int8_t[conv2.output_channels*conv2.in_channels*conv2.ksize*conv2.ksize]; 
    int16_t m0 [1];
    load_int8_data(conv2.weights ,  data_dir+"/conv2/" +"w_q.bin" , conv2.output_channels*conv2.in_channels*conv2.ksize*conv2.ksize); 
    load_int16_data(m0 , data_dir+"/conv2/" +"m_0.bin" , 1);
    load_int32_data(conv2.bias ,  data_dir+"/conv2/" +"b_q.bin" , conv2.output_channels);
    conv2.data_update(&conv2.weights[0],&conv2.bias[0],* m0);

    int32_t *buffer1;
    buffer1 = new int32_t[conv2.out_shape[0]*conv2.out_shape[1]*conv2.out_shape[2]];
    fill_n(buffer1,conv2.out_shape[0]*conv2.out_shape[1]*conv2.out_shape[2],0);
    conv2.forward(Out1,buffer1);
    delete [] Out1;
    tensor2int8(buffer1,out,conv2.out_shape[0],conv2.out_shape[1],conv2.out_shape[2]);
    delete [] buffer1;

}

///////////////////////////////

void up_sample::init(unsigned int in_shape[3])
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape[i];
    }   
    this->conv1.init(this->in_shape,this->in_shape[0]/2,1,1);
    this->upsample1.init(conv1.conv.out_shape);

}

void up_sample::forward(data_t *x,data_t *out)
{
    string data_dir=syn_int8m16_dir + "upsample/";
    conv1.data_update(data_dir+"/conv1/"); 
    int8_t *Out1;
    Out1 = new int8_t[conv1.conv.out_shape[0]*conv1.conv.out_shape[1]*conv1.conv.out_shape[2]];
    conv1.forward(x,Out1);
       
    
    fill_n(out,upsample1.out_shape[0]*upsample1.out_shape[1]*upsample1.out_shape[2],0);
    upsample1.forward(Out1,out);
    delete [] Out1;

}

////////////////////////////////

void yolo_body::init()
{
    backbone.init();

    unsigned int yolohead_p4_in_shape[3] ={384,26,26};
    yolohead_p4.init(yolohead_p4_in_shape,256,75,"yolo_headP4");
    unsigned int yolohead_p5_in_shape[3] ={256,13,13};
    yolohead_p5.init(yolohead_p5_in_shape,256,75,"yolo_headP5");
    unsigned int up_sample0_in_shape[3] ={256,13,13};
    up_sample0.init(up_sample0_in_shape);
    unsigned int conv_for_p5_in_shape[3] ={512,13,13};
    conv_for_p5.init(conv_for_p5_in_shape,256,1,1);


}

void yolo_body::forward(data_t *x,data_t *out0,data_t *out1)
{
    string data_dir=syn_int8m16_dir ;
    //backbone
    int8_t *Out0;
    int8_t *Out1;
    Out0 = new int8_t[backbone.resblock_body3.conv4.conv.in_shape[0]*backbone.resblock_body3.conv4.conv.in_shape[1]*backbone.resblock_body3.conv4.conv.in_shape[2]];
    Out1 = new int8_t[backbone.conv3.conv.out_shape[0]*backbone.conv3.conv.out_shape[1]*backbone.conv3.conv.out_shape[2]];
    backbone.forward(x,Out0,Out1);
    //FPN
    //conv_for_p5
    int8_t *P5;
    P5 = new int8_t[conv_for_p5.conv.out_shape[0]*conv_for_p5.conv.out_shape[1]*conv_for_p5.conv.out_shape[2]];
    conv_for_p5.data_update(data_dir+"/conv_for_P5/"); 
    conv_for_p5.forward(Out1,P5);
    delete [] Out1;

    //up_sample0
    int8_t *P5_Upsample;
    P5_Upsample = new int8_t[up_sample0.upsample1.out_shape[0]*up_sample0.upsample1.out_shape[1]*up_sample0.upsample1.out_shape[2]];
    up_sample0.forward(P5,P5_Upsample);
    
    //YOLO HEAD
    //yolohead_p4
    int8_t *P4;
    P4 = new int8_t[26*26*384];
    concatenate_P5_feat1(P5_Upsample,Out0,P4);
    delete [] P5_Upsample;
    delete [] Out0;
    yolohead_p4.forward(P4,out1);
    delete [] P4;
    //yolohead_p5
    yolohead_p5.forward(P5,out0);
    delete [] P5;

}








////////////////////////////////


data_t*** padArray(const data_t* originalArray,int chi, int originalRows, int originalCols) {  
    int newRows = originalRows + 1;  
    int newCols = originalCols + 1;  
  
    // 分配指向指针数组的指针数组  
    data_t*** paddedArray = new data_t**[chi];  
    for (int k = 0; k < chi; ++k) {  
        // 分配指向行数组的指针数组  
        paddedArray[k] = new data_t*[newRows];  
        for (int i = 0; i < newRows; ++i) {  
            // 分配每行的数据数组  
            paddedArray[k][i] = new data_t[newCols];  
        }  
    }  
  
    // 初始化填充区域  
    /*for (int k = 0; k < chi; ++k) {  
        for (int i = 0; i < newRows; ++i) {  
            for (int j = 0; j < newCols; ++j) {  
                if (i == 0 || i == newRows - 1 || j == 0 || j == newCols - 1) {  
                    paddedArray[k][i][j] = 0; // 填充边界  
                } else {  
                    paddedArray[k][i][j] = data_t(); // 初始化内部元素，假设data_t有默认构造函数  
                }  
            }  
        }  
    }  */
    for (int k = 0; k < chi; ++k) {  
        for (int i = 0; i < newRows; ++i) {  
            for (int j = 0; j < newCols; ++j) {  
                if (i == 0 ||  j == 0 ) {  
                    paddedArray[k][i][j] = 0; // 填充边界  
                } else {  
                    paddedArray[k][i][j] = data_t(); // 初始化内部元素，假设data_t有默认构造函数  
                }  
            }  
        }  
    }
    // 复制原始数据到填充数组  
    int index = 0;  
    for (int k = 0; k < chi; ++k) {  
        for (int i = 1; i < newRows ; ++i) {  
            for (int j = 1; j < newCols ; ++j) {  
                paddedArray[k][i][j] = originalArray[index++];  
            }  
        }  
    }  
  
    return paddedArray;  
}

data_t * flatten_3d(data_t *** arr,int size2,int size1,int size0)
{
    data_t * arr_1d;
    arr_1d= new  data_t[size2*size1*size0];
    int index = 0;  
    for (int k = 0; k < size2; k++) {  
        for (int i = 0; i < size1 ; i++) {  
            for (int j = 0; j < size0 ; j++) {  
                arr_1d[index++] = arr[k][i][j];  
                std::cout << arr_1d[index - 1] << " "; 
            } 
            std::cout << std::endl;  
        }  
        std::cout << std::endl; 
    }  
    return arr_1d;
}

void tensor2int8(int32_t * in,int8_t * out,int size0,int size1,int size2)
{
    for (int i=0;i<size0;i++)
        for ( int j = 0;j<size1;j++)
            for ( int k = 0; k < size2 ; k ++ )
                {
                    //in[i][j][k]
                    if ( in[(i*size1+j)*size2+k] >127)
                        out[(i*size1+j)*size2+k] = 127 ;
                    else if ( in[(i*size1+j)*size2+k] <-128 )
                        out[(i*size1+j)*size2+k] = -128 ;
                    else 
                        out[(i*size1+j)*size2+k] = static_cast<int8_t>(in[(i*size1+j)*size2+k]);

                }
}

void concatenate(int8_t * tensor0,int8_t * tensor1, int8_t * out,int channels,int width)
{
    
    for (int i=0;i<channels;i++)
        for (int j=0;j<width;j++)
            for (int k=0;k<width;k++)
            {
                out[(i*width+j)*width+k]=tensor0[(i*width+j)*width+k];
            }
    for (int i=0;i<channels;i++)
        for (int j=0;j<width;j++)
            for (int k=0;k<width;k++)
            {
                out[(i*width+j)*width+k+channels*width*width-1]=tensor1[(i*width+j)*width+k];
            }    

    
}

void concatenate_P5_feat1(int8_t * P5_Upsample,int8_t * feat1, int8_t * out)
{
    
    for (int i=0;i<256;i++)
        for (int j=0;j<26;j++)
            for (int k=0;k<26;k++)
            {
                out[(i*26+j)*26+k]=P5_Upsample[(i*26+j)*26+k];
            }
    for (int i=0;i<128;i++)
        for (int j=0;j<26;j++)
            for (int k=0;k<26;k++)
            {
                out[(i*26+j)*26+k+256*26*26-1]=feat1[(i*26+j)*26+k];
            }  
}

///////////////////////






/*
int main() {  
    
    int8_t *in;
    in=new int8_t[3*416*416]; 
    load_int8_data(in ,  syn_data_dir +"in_q.bin", 3*416*416); 
    cout << "start!" <<endl;
    
    yolo_body my_yolo_tst;
    my_yolo_tst.init();
    int8_t *Out0;
    int8_t *Out1;
    Out0 = new int8_t[my_yolo_tst.yolohead_p5.conv2.out_shape[0]*my_yolo_tst.yolohead_p5.conv2.out_shape[1]*my_yolo_tst.yolohead_p5.conv2.out_shape[2]];
    Out1 = new int8_t[my_yolo_tst.yolohead_p4.conv2.out_shape[0]*my_yolo_tst.yolohead_p4.conv2.out_shape[1]*my_yolo_tst.yolohead_p4.conv2.out_shape[2]];
    my_yolo_tst.forward(&in[0],Out0,Out1);
    
    save_int8_data(Out0, syn_data_dir + "yolo_out0_q8.bin" , my_yolo_tst.yolohead_p5.conv2.out_shape[0]*my_yolo_tst.yolohead_p5.conv2.out_shape[1]*my_yolo_tst.yolohead_p5.conv2.out_shape[2]);
    save_int8_data(Out1, syn_data_dir + "yolo_out1_q8.bin" , my_yolo_tst.yolohead_p4.conv2.out_shape[0]*my_yolo_tst.yolohead_p4.conv2.out_shape[1]*my_yolo_tst.yolohead_p4.conv2.out_shape[2]);

    delete [] Out0;
    delete [] Out1;
    
      
    return 0;  
}
*/

void tensor_int32_init(int32_t * tensor,int size )
{
    for (int i=0;i<size;i++)
    {
        tensor[i]=0;
    }
}
void tensor_int8_init(int8_t * tensor,int size )
{
    for (int i=0;i<size;i++)
    {
        tensor[i]=0;
    }
}

void print_out_tensor(int32_t * out ,int cho,int width)
{
    for (int i =0;i<cho;i++)
    {
        for (int j=0;j<width;j++)
        {
            for (int k=0;k<width;k++)
            {
                std::cout<<out[(i*width+j)*width+k]<<" ";
            }
        std::cout<<endl;
        }
    std::cout<<endl;
    }

}


int load_int8_data(int8_t *buffer , const string dir,int size)
{
	int success=0;

	success = read_int8(dir, buffer, size);
    return success;
}

int load_int16_data(int16_t *buffer , const string dir,int size)
{
	int success=0;

	success = read_int16(dir, buffer, size);
    return success;
}

int load_int32_data(int32_t *buffer , const string dir,int  size)
{
	int success=0;

	success = read_int32(dir, buffer, size);
    return success;
}

int save_int32_data(int32_t *buffer ,const string dir ,int size)
{
	int success=0;

	success = write_int32(dir, buffer, size);
    return success;
}

int save_int8_data(int8_t *buffer ,const string dir ,int size)
{
	int success=0;

	success = write_int8(dir, buffer, size);
    return success;
}


#include <iostream>
#include <vector>   
#include <cstring>
#include <cmath>
#include <fstream>  
#include <string>  
using namespace std;



const string syn_data_dir = "/home/derui/work/Py_proj/yolo_tiny_v4_baseline/hw_tst/conv1_relu0125/int8_m16/";

int load_int8_data(int8_t *buffer , const string dir,streamsize size)
{
    std::ifstream file(dir, std::ios::binary | std::ios::in);  
      
    if (!file) {  
        std::cerr << "无法打开文件" << std::endl;  
        return 1;  
    }  

      
    if (!file.read(reinterpret_cast<char*>(buffer), size)) {  

        std::cerr << "读取文件时出错" << std::endl;  
        return 1;  
    } 
    // 关闭文件  
    file.close(); 
    return 0;
}

int load_int16_data(int16_t *buffer , const string dir,streamsize size)
{
    std::ifstream file(dir, std::ios::binary | std::ios::in);  
      
    if (!file) {  
        std::cerr << "无法打开文件" << std::endl;  
        return 1;  
    }  

      
    if (!file.read(reinterpret_cast<char*>(buffer), size)) {  

        std::cerr << "读取文件时出错" << std::endl;  
        return 1;  
    } 
    // 关闭文件  
    file.close(); 
    return 0;
}

int load_int32_data(int32_t *buffer , const string dir,streamsize size)
{
    std::ifstream file(dir, std::ios::binary | std::ios::in);  
      
    if (!file) {  
        std::cerr << "无法打开文件" << std::endl;  
        return 1;  
    }  

      
    if (!file.read(reinterpret_cast<char*>(buffer), size)) {  

        std::cerr << "读取文件时出错" << std::endl;  
        return 1;  
    } 
    // 关闭文件  
    file.close(); 
    return 0;
}








/*-------------env----------------*/


#define K 3
#define Pr 208
#define Pc 208
#define Pof 32
#define Pif 3
#define S 2
#define Cho_max 512

#define Tk 4
#define Tc 4
#define Tp 8

//kloop_max=ceil(Pof/Tk)
#define K_LOOP_MAX 4
//ploop_max=ceil(Pr*Pc/Tp)
#define P_LOOP_MAX 32
//cloop_max=ceil(float(Pif)/float(Tc))
#define C_LOOP_MAX 1

#define S_RELU 4

void loop_1(int32_t   bias[Cho_max],int8_t *In_addr,int8_t *W_addr,int8_t *Out_addr,int16_t  M0,int32_t k_loop_max,int32_t p_loop_max,
	int32_t c_loop_max,int32_t out_map_height,int32_t out_map_width,int32_t out_map_cho,int32_t in_map_chi,
	int32_t in_map_height,int32_t in_map_width,int32_t ksize,int32_t stride ,int32_t activation_enable,int32_t S_relu);
void DMA_1(int8_t   in_L2[Tp][Tc],int8_t weights_L1[Tk][Tc],int8_t *In_addr,int8_t *W_addr,int c,int ky,int kx,int k,int p,
	int32_t out_map_cho,int32_t in_map_chi,int32_t in_map_height,int32_t in_map_width,int32_t ksize,int32_t stride,int32_t pad);
void Loop_2(int8_t   in_L2[Tp][Tc],int8_t weights_L1[Tk][Tc],int32_t   bias[Cho_max],int32_t   out[Tp][Tk],int k,int p,
	int8_t *Out_addr,int8_t *In_addr,int8_t *W_addr,int16_t  M0,int32_t c_loop_max,int32_t out_map_height,int32_t out_map_width,
	int32_t out_map_cho,int32_t in_map_chi,int32_t in_map_height,int32_t in_map_width,int32_t ksize,int32_t stride,int32_t pad,int32_t activation_enable,int32_t S_relu);
void PE_Top(int8_t   in_L2[Tp][Tc],int8_t weights_L1[Tk][Tc],int32_t   bias[Cho_max],int32_t   out[Tp][Tk],int c,int ky,int kx,int k,int p,int pp);
void DMA_2(int8_t   in_L2[Tp][Tc],int8_t in_L1[Tc],int pp);
void PE(int32_t out[Tp][Tk],int8_t in_L1[Tc],int8_t weights_L1[Tk][Tc],int pp);
void Conv_deal(int8_t *In_addr,int8_t *W_addr,int32_t *B_addr,int8_t *Out_addr,int32_t *Param);
void Bias_load(int32_t *B_addr,int32_t *bias,int32_t out_map_cho);
void base_conv(int8_t *In_addr,int8_t *W_addr,int32_t *B_addr,int8_t *Out_addr,int32_t *Param);

void base_conv(int8_t *In_addr,int8_t *W_addr,int32_t *B_addr,int8_t *Out_addr,int32_t *Param)
{

	Conv_deal(In_addr,W_addr,B_addr,Out_addr,Param);
}


void Bias_load(int32_t *B_addr,int32_t *bias,int32_t out_map_cho)
{
    for( int i=0;i<out_map_cho;i++)
    {
        bias[i]=B_addr[i];
    }
}

void Conv_deal(int8_t *In_addr,int8_t *W_addr,int32_t *B_addr,int8_t *Out_addr,int32_t *Param)
{
	int32_t   bias[Cho_max]={0};
	static int16_t M0 ;
    static int32_t k_loop_max,p_loop_max, c_loop_max, out_map_height, out_map_width, out_map_cho;
    static int32_t in_map_chi, in_map_height, in_map_width;
	static int32_t ksize,stride;
	static int32_t activation_enable;
	static int16_t S_relu;
	static int32_t mul_factor;
	static int32_t shift_factor;

	M0 = static_cast<int16_t>(Param[0]);   
    k_loop_max		=Param[1];
    p_loop_max		=Param[2];
    c_loop_max		=Param[3];
    out_map_height	=Param[4];
    out_map_width 	=Param[5];
    out_map_cho		=Param[6];
    in_map_chi		=Param[7];
    in_map_height 	=Param[8];
    in_map_width	=Param[9];
	ksize			=Param[10];
	stride			=Param[11];
	activation_enable=Param[12];
    S_relu			=static_cast<int16_t>(Param[13]);

	

	Bias_load(B_addr,bias, out_map_cho);
    loop_1(   bias,In_addr, W_addr, Out_addr,  M0, k_loop_max, p_loop_max, c_loop_max,out_map_height, out_map_width, out_map_cho, in_map_chi, in_map_height, in_map_width, ksize, stride,activation_enable,S_relu);

}

void PE(int32_t out[Tp][Tk],int8_t in_L1[Tc],int8_t weights_L1[Tk][Tc],int pp)
{
	for(int kk=0;kk<Tk;kk++)
		for(int cc=0;cc<Tc;cc++)
		{
			out[pp][kk]+=in_L1[cc]*weights_L1[kk][cc];
			//printf("[%d][%d] In:%d,W:%d,sum:%d \n",pp,kk,in_L1[cc],weights_L1[kk][cc],out[pp][kk]);
		}
}

void DMA_2(int8_t   in_L2[Tp][Tc],int8_t in_L1[Tc],int pp)
{
    for(int cc=0;cc<Tc;cc++)
    {
		in_L1[cc]=in_L2[pp][cc];
    }
}

void PE_Top(int8_t   in_L2[Tp][Tc],int8_t weights_L1[Tk][Tc],int32_t   bias[Cho_max],int32_t   out[Tp][Tk],
		int c,int ky,int kx,int k,int p,int pp)
{
	int8_t in_L1[Tc]={0};
	DMA_2( in_L2,in_L1,pp);
	PE(out,in_L1,weights_L1,pp);
}

void Loop_2(int8_t   in_L2[Tp][Tc],int8_t weights_L1[Tk][Tc],int32_t   bias[Cho_max],int32_t   out[Tp][Tk],
    int k,int p,int8_t *Out_addr,int8_t *In_addr,int8_t *W_addr,int16_t  M0,int32_t c_loop_max,int32_t out_map_height,
    int32_t out_map_width,int32_t out_map_cho,int32_t in_map_chi,int32_t in_map_height,int32_t in_map_width,int32_t ksize,
	int32_t stride,int32_t pad,int32_t activation_enable,int32_t S_relu )
{
	int32_t Out=0;		
	for(int c=0;c<c_loop_max;c++)
		for(int ky=0;ky<ksize;ky++)
			for(int kx=0;kx<ksize;kx++)
				{
                    DMA_1(   in_L2, weights_L1, In_addr, W_addr, c, ky, kx, k, p, out_map_cho, in_map_chi, in_map_height, in_map_width,ksize,stride,pad);
                    for(int pp=0;pp<Tp;pp++)
                    {
                        if(p*Tp+pp<Pr*Pc)
				            PE_Top( in_L2, weights_L1, bias, out, c, ky, kx, k, p,pp);
                    }
				}
	for(int pp=0;pp<Tp;pp++)
	{
		int h=(p*Tp+pp)/Pr;
		int w=(p*Tp+pp)%Pr;
		for(int kk=0;kk<Tk;kk++)
        {
			if ( (((k*Tk+kk)*out_map_height+h)*out_map_width+w)==0 )
				{
					printf("Out[0]: %d ",out[pp][kk]);
					printf("bias[0]: %d ",bias[k*Tk+kk]);

				}
			Out=((out[pp][kk]+bias[k*Tk+kk])*M0)>>15;
			if ( (((k*Tk+kk)*out_map_height+h)*out_map_width+w)==0 )
				{printf("Out[0]: %d ",Out);}
			if(activation_enable)
			{
				if(Out>=0)
				{
					Out=(Out * S_relu)>>14;
				}
				else
				{
					Out=(Out * S_relu)>>17;
				}
				if ( (((k*Tk+kk)*out_map_height+h)*out_map_width+w)==0 )
				{printf("Out[0]: %d ",Out);}
			}
			
			Out_addr[((k*Tk+kk)*out_map_height+h)*out_map_width+w]=int8_t (Out);
				
        }
	}
}

void DMA_1(int8_t   in_L2[Tp][Tc],int8_t weights_L1[Tk][Tc],int8_t *In_addr,int8_t *W_addr,int c,int ky,int kx,int k,int p,
    int32_t out_map_cho,int32_t in_map_chi,int32_t in_map_height,int32_t in_map_width,int32_t ksize,int32_t stride,int32_t pad)
{
	for(int pp=0;pp<Tp;pp++)
	{
		if(p*Tp+pp<Pr*Pc)
		{
			int h=(p*Tp+pp)/Pr;
			int w=(p*Tp+pp)%Pr;
			for(int kk=0;kk<Tk;kk++)
				for(int cc=0;cc<Tc;cc++)
					{
						if( ((c*Tc+cc)<in_map_chi)&&(p*Tp+pp<Pr*Pc)   &&    ((ksize==1) ||  (((h*stride+ky)!=0) && ((h*stride+ky)!=(in_map_height+1)) &&((w*stride+kx)!=0) && ((w*stride+kx)!=(in_map_width+1)))) )
						{

							int in_offset=((c*Tc+cc)*in_map_height+h*stride+ky-pad)*in_map_width+w*stride+kx-pad;
							in_L2[pp][cc]	    =  In_addr[in_offset];   //In_addr[c*Tc+cc][h*S+ky][w*S+kx];
							int w_offset=(((k*Tk+kk)*in_map_chi+c*Tc+cc)*ksize+ky)*ksize+kx;
							weights_L1[kk][cc]  =  W_addr[w_offset];                  //W_addr[k*Tk+kk][c*Tc+cc][ky][kx];
						}
						else if(	(ksize!=1)	&&(((h*stride+ky)==0) || ((h*stride+ky)==(in_map_height+1)) || ((w*stride+kx)==0) || ((w*stride+kx)==(in_map_width+1)))	)
						{
							in_L2[pp][cc]=0;
							int w_offset=(((k*Tk+kk)*in_map_chi+c*Tc+cc)*ksize+ky)*ksize+kx;
							weights_L1[kk][cc]  =  W_addr[w_offset]; 
						}
						else
						{
							in_L2[pp][cc]=0;
							weights_L1[kk][cc]=0;
						}
					}
		}
	}
}

void loop_1(int32_t   bias[Cho_max],int8_t *In_addr,int8_t *W_addr,int8_t *Out_addr,int16_t  M0,int32_t k_loop_max,int32_t p_loop_max,int32_t c_loop_max,
    int32_t out_map_height,int32_t out_map_width,int32_t out_map_cho,int32_t in_map_chi,int32_t in_map_height,int32_t in_map_width,
	int32_t ksize,int32_t stride,int32_t activation_enable,int32_t S_relu)
{
    int8_t    in_L2[Tp][Tc]={0};                //In[Pif][S*Pr+K-S][S*Pc+K-S];
    int8_t    weights_L1[Tk][Tc]={0};           //W[Pof][Pif][K][K];
             
    int32_t pad=(ksize==1)?0:1;

	cout<<M0				<<endl;
	cout<<k_loop_max		<<endl;
	cout<<p_loop_max		<<endl;
	cout<<c_loop_max		<<endl;
	cout<<out_map_height	<<endl;
	cout<<out_map_width 	<<endl;
	cout<<out_map_cho		<<endl;
	cout<<in_map_chi		<<endl;
	cout<<in_map_height 	<<endl;
    cout<<in_map_width	 	<<endl;
	cout<<ksize	 	<<endl;
	cout<<stride	 	<<endl;
	cout<<activation_enable	 	<<endl;
	cout<<S_relu	 	<<endl;

    for(int k=0;k<k_loop_max;k++)
		for(int p=0;p<p_loop_max;p++)
		{
			int32_t   out[Tp][Tk]={0};                  //Out[Pof][Pr][Pc]; 
            Loop_2( in_L2, weights_L1,   bias,   out,k, p,Out_addr,In_addr, W_addr,  M0, c_loop_max,out_map_height, out_map_width, out_map_cho, in_map_chi, in_map_height, in_map_width, ksize, stride,pad,activation_enable, S_relu);

        }
}
///////////////tb:

void data_load_bin(int8_t *In,int8_t *W,int32_t *Bias,int32_t Param[14],int8_t *Out_sw)
{
    load_int8_data(In ,  syn_data_dir +"in_q.bin", 3*416*416); 
	load_int8_data(W ,  syn_data_dir +"w_q.bin" , Pof*Pif*K*K); 
	load_int32_data(Bias ,  syn_data_dir +"b_q.bin" , streamsize (4*Pof)); 
	int16_t m0_temp[1]={0};
    load_int16_data(m0_temp , syn_data_dir +"m_0.bin" , 2);
	int16_t s0[1]={0};
	load_int16_data(s0 , syn_data_dir +"S_relu_q.bin" , 2);

    Param[0]=int32_t(m0_temp[0])	;	//M0=int16_t(Param[0]);   
    Param[1]=ceil(Pof/Tk)	;	//k_loop_max=Param[1];//kloop_max=ceil(Pof/Tk)
    Param[2]=ceil(Pr*Pc/float(Tp))	;	//p_loop_max=Param[2];//ploop_max=ceil(Pr*Pc/Tp)
    Param[3]=ceil(float(Pif)/float(Tc))	;	//c_loop_max=Param[3];//cloop_max=ceil(float(Pif)/float(Tc))
    Param[4]=Pr	;	//out_map_height=Param[4];
    Param[5]=Pc	;	//out_map_width =Param[5];
    Param[6]=Pof	;	//out_map_cho=Param[6];
    Param[7]=Pif	;	//in_map_chi=Param[7];
    Param[8]=416	;	//in_map_height =Param[8];
    Param[9]=416	;	//in_map_width=Param[9];
	Param[10]=K	;//ksize
	Param[11]=S	;//stride
	Param[12]=1 ;//activation_enable
	Param[13]= s0[0];

	load_int8_data(Out_sw ,  syn_data_dir +"acc_Out_q.bin" , Pof*Pr*Pc); 

}

void data_gen(int8_t In[Pif][S*Pr+K-S][S*Pc+K-S],int8_t W[Pof][Pif][K][K],int32_t Bias[Pof],int32_t Param[14])
{
	for(int i=0;i<Pof;i++)
		for(int j=0;j<Pif;j++)
			for(int k=0;k<K;k++)
				for(int l=0;l<K;l++)
				{
					W[i][j][k][l]=i+j+k+l;
				}
	for(int i=0;i<Pif;i++)
		for(int j=0;j<(S*Pr+K-S);j++)
			for(int k=0;k<(S*Pc+K-S);k++)
			{
				In[i][j][k]=i+j+k;
			}
	for(int i=0;i<Pof;i++)
	{
		Bias[i]=i;
	}
    Param[0]=1	;	//M0=int16_t(Param[0]);   
    Param[1]=K_LOOP_MAX	;	//k_loop_max=Param[1];
    Param[2]=P_LOOP_MAX	;	//p_loop_max=Param[2];
    Param[3]=C_LOOP_MAX	;	//c_loop_max=Param[3];
    Param[4]=Pr	;	//out_map_height=Param[4];
    Param[5]=Pc	;	//out_map_width =Param[5];
    Param[6]=Pof	;	//out_map_cho=Param[6];
    Param[7]=Pif	;	//in_map_chi=Param[7];
    Param[8]=S*Pr+K-S	;	//in_map_height =Param[8];
    Param[9]=S*Pc+K-S	;	//in_map_width=Param[9];
	Param[10]=K	;//ksize
	Param[11]=S	;//stride
	Param[12]=0 ;//activation_enable
	Param[13]=S_RELU ;
}

void sw_out(int8_t Out_sw[Pof][Pr][Pc],int8_t In[Pif][S*Pr+K-S][S*Pc+K-S],int8_t W[Pof][Pif][K][K],int32_t Bias[Pof])
{
	int32_t out_temp[Pof][Pr][Pc]={0};
	Kernel_Row:
	for(int kr=0;kr<K;kr++)
	{
		Kernel_Col:
		for(int kc=0;kc<K;kc++)
		{
			Row:
			for(int r=0;r<Pr;r++)
			{
				Column:
				for(int c=0;c<Pc;c++)
				{
					Out_channel:
					for(int out=0;out<Pof;out++)
					{
						in_channel:
						for(int in=0;in<Pif;in++)
						{
							if(K==3)
								if(	((S*r+kr)!=0)&&((S*c+kc)!=0)&&((S*c+kc)!=(S*Pc+K-S+1))&&((S*r+kr)!=(S*Pr+K-S+1))	)
									{out_temp[out][r][c]+=In[in][S*r+kr-1][S*c+kc-1]*W[out][in][kr][kc];}
								else
								{}
							else
								out_temp[out][r][c]+=In[in][S*r+kr][S*c+kc]*W[out][in][kr][kc];

							if((kr==K-1)&&(kc==K-1)&&(in==Pif-1))
							{
								out_temp[out][r][c]=out_temp[out][r][c]+Bias[out];
								Out_sw[out][r][c] = int8_t(out_temp[out][r][c]);
								
								
							}
								
						}
						
					}
				}
			}
		}
	}
}
int test_print(int8_t Out_sw[Pof][Pr][Pc],int8_t Out_hw[Pof][Pr][Pc])
{
	int err_sum=0;
	printf("Out_sw \n");
	
	printf("Out[0]: %d ",Out_sw[0][0][0]);
	for(int i=0;i<Pof;i++)
	{
		for(int j=0;j<Pr;j++)
		{
			for(int k=0;k<Pc;k++)
			{
				//printf("%d ",Out_sw[i][j][k]);

				if(Out_sw[i][j][k]!=Out_hw[i][j][k])
				{
					//printf("Out_sw[%d][%d][%d]=%d    while Out_hw=%d ",i,j,k,Out_sw[i][j][k],Out_hw[i][j][k]);
					err_sum++;
				}

			}
			//printf("\n");
		}
		//printf("\n");
	}

	//printf("Out_hw \n");

	/*	for(int i=0;i<Pof;i++)
	{
		for(int j=0;j<Pr;j++)
		{
			for(int k=0;k<Pc;k++)
			{
				printf("%d ",Out_hw[i][j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}*/
	//printf("Out_sw[%d][%d][%d]=%d    while Out_hw=%d \n",0,0,0,Out_sw[0][0][0],Out_hw[0][0][0]);
	return err_sum;
}


int8_t In[Pif][416][416];
int8_t W[Pof][Pif][K][K];
int32_t Bias[Pof];
int8_t Out_sw[Pof][Pr][Pc]={0};
int8_t Out_hw[Pof][Pr][Pc]={0};
int32_t Param[14];

int main() { 
	int err=0;
	int8_t *Out_p=&Out_hw[0][0][0];
	int8_t *In_p=&In[0][0][0];
	int8_t *W_p=&W[0][0][0][0];
	int32_t *Bias_p=&Bias[0];
	data_load_bin( In_p,W_p, Bias, Param,&Out_sw[0][0][0]);

	//data_gen(In,W,Bias,Param);
	//sw_out(Out_sw,In,W,Bias);
	base_conv(In_p,W_p,Bias_p,Out_p,Param);
	printf("start\n");
	err=test_print(Out_sw,Out_hw);
	printf("finish\n");
	printf("err:%d",err);


    return 0;

}
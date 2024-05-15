#include <iostream>
#include <vector>   
#include <cstring>
using namespace std;


#define K 2
#define S 2
#define Pr 16
#define Pc 16
#define Pof 3

#define Cho_max 512

void base_maxpool(int8_t *In_addr,int8_t *Out_addr,int32_t *Param);
void Maxpool_deal(int8_t *In_addr,int8_t *Out_addr,int32_t *Param);
void load_in_buf(int8_t *In_addr ,int8_t *in_buf,int32_t in_map_width,int32_t cho);
void horizon_maxpool(int8_t *in_buf ,int8_t *out_temp,int32_t out_map_width);
void vertical_maxpool(int8_t *out_temp,int8_t *out_buf,int32_t out_map_width);
void load_out(int8_t *Out_addr ,int8_t *out_buf,int32_t out_map_width,int32_t cho);



void base_maxpool(int8_t *In_addr,int8_t *Out_addr,int32_t *Param)
{

	Maxpool_deal(In_addr,Out_addr,Param);
}

void Maxpool_deal(int8_t *In_addr,int8_t *Out_addr,int32_t *Param)
{
	static int32_t  out_map_width, out_map_cho;   
    out_map_width	=Param[0];
    out_map_cho		=Param[1];

	int8_t in_buf[104*104]={0};
	int8_t out_temp[104*52]={0};
	int8_t out_buf[52*52]={0};

	printf("out_map_width:  %d \n",out_map_width);
	printf("out_map_cho:  %d \n",out_map_cho);
	for(int cho=0;cho<out_map_cho;cho++)
	{
		load_in_buf(In_addr ,in_buf,2*out_map_width,cho);
		horizon_maxpool( in_buf , out_temp, out_map_width);
		vertical_maxpool( out_temp, out_buf, out_map_width);
		load_out(Out_addr ,out_buf, out_map_width, cho);
	}
}

void load_in_buf(int8_t *In_addr ,int8_t *in_buf,int32_t in_map_width,int32_t cho)
{
	for (int i=0;i<in_map_width;i++)
		for(int j=0;j<in_map_width;j++)
		{
			in_buf[i*in_map_width+j]=In_addr[(cho*in_map_width+i)*in_map_width+j];

		}
}

void horizon_maxpool(int8_t *in_buf ,int8_t *out_temp,int32_t out_map_width)
{
	for(int i=0;i<2*out_map_width*out_map_width;i++)
	{
		if(in_buf[2*i]>in_buf[2*i+1])
		{
			out_temp[i]=in_buf[2*i];
		}
		else
		{
			out_temp[i]=in_buf[2*i+1];
		}
	}
}

void vertical_maxpool(int8_t *out_temp,int8_t *out_buf,int32_t out_map_width)
{
	for(int i=0;i<out_map_width;i++)
	{
		for(int j=0;j<out_map_width;j++)
		{
			if(out_temp[(2*j)*out_map_width+i]>out_temp[(2*j+1)*out_map_width+i])	//[2j][i]  [2j+1][i]
			{
				out_buf[j*out_map_width+i]=out_temp[(2*j)*out_map_width+i];
			}
			else
			{
				out_buf[j*out_map_width+i]=out_temp[(2*j+1)*out_map_width+i];
			}
		}
	}
}
void load_out(int8_t *Out_addr ,int8_t *out_buf,int32_t out_map_width,int32_t cho)
{
	for (int i=0;i<out_map_width;i++)
		for(int j=0;j<out_map_width;j++)
		{
			Out_addr[(cho*out_map_width+i)*out_map_width+j]=out_buf[i*out_map_width+j];
		}
}

/*-------tb--------*/


void data_gen(int8_t In[Pof][2*Pr][2*Pc],int32_t Param[2])
{
	for(int i=0;i<Pof;i++)
		for(int j=0;j<2*Pr;j++)
			for(int k=0;k<2*Pc;k++)
			{
				In[i][j][k]=i+j+k;
			}

    Param[0]=	Pr;	  //  out_map_width	 
    Param[1]=	Pof;	  //  out_map_cho		

}



int8_t find_max(int8_t x0,int8_t x1,int8_t x2,int8_t x3)
{
    int8_t max=x0;
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

void sw_out(int8_t *Out_sw,int8_t *In)
{
	for(int r=0;r<Pr;r++)
	{
		//Column:
		for(int c=0;c<Pc;c++)
		{
		    //Out_channel:
			for(int o=0;o<Pof;o++)
			{
                int x_offset0=(o*2*Pr+2*r)*2*Pr+2*c;
                int x_offset1=(o*2*Pr+2*r)*2*Pr+2*c+1;
                int x_offset2=(o*2*Pr+2*r+1)*2*Pr+2*c;
                int x_offset3=(o*2*Pr+2*r+1)*2*Pr+2*c+1;

                int out_offset=(o*Pr+r)*Pr+c;
              
                *(Out_sw+out_offset) = find_max(*(In+x_offset0) ,*(In+x_offset1),*(In+x_offset2),*(In+x_offset3)) ;
			}
		}
	}
}

int test_print(int8_t Out_sw[Pof][Pr][Pc],int8_t Out_hw[Pof][Pr][Pc])
{
	int err_sum=0;
	printf("Out_sw \n");
	for(int i=0;i<Pof;i++)
	{
		for(int j=0;j<Pr;j++)
		{
			for(int k=0;k<Pc;k++)
			{
				printf("%d ",Out_sw[i][j][k]);

				if(Out_sw[i][j][k]!=Out_hw[i][j][k])
				{
					//printf("Out_sw[%d][%d][%d]=%d    while Out_hw=%d ",i,j,k,Out_sw[i][j][k],Out_hw[i][j][k]);
					err_sum++;
				}

			}
			printf("\n");
		}
		printf("\n");
	}

	printf("Out_hw \n");

	for(int i=0;i<Pof;i++)
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
	}
	
	return err_sum;
}

int8_t In[Pof][2*Pr][2*Pc];
int8_t Out_sw[Pof][Pr][Pc]={0};
int8_t Out_hw[Pof][Pr][Pc]={0};
int32_t Param[2];

int main() { 
	int err=0;
	int8_t *Out_p=&Out_hw[0][0][0];
	int8_t *In_p=&In[0][0][0];

	data_gen(In,Param);
	sw_out(&Out_sw[0][0][0],In_p);
	base_maxpool(In_p, &Out_hw[0][0][0], Param);
	printf("start\n");
	err=test_print(Out_sw,Out_hw);
	printf("finish\n");
	printf("err:%d",err);


    return 0;

}
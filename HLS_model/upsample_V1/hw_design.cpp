#include <iostream>
#include <vector>   
#include <cstring>
using namespace std;

#define Pif 128
#define Pic 13
#define Pir 13
#define Pr 26
#define Pc 26
#define Pof 128



void base_upsample(int8_t *In_addr,int8_t *Out_addr);
void Upsample_deal(int8_t *In_addr,int8_t *Out_addr);
void load_in_buf(int8_t *In_addr ,int8_t *in_buf,int32_t cho);
void horizon_upsample(int8_t *in_buf ,int8_t *out_temp);
void vertical_upsample(int8_t *out_temp,int8_t *out_buf);
void load_out(int8_t *Out_addr ,int8_t *out_buf,int32_t cho);



void base_upsample(int8_t *In_addr,int8_t *Out_addr)
{

	Upsample_deal(In_addr,Out_addr);
}

void Upsample_deal(int8_t *In_addr,int8_t *Out_addr)
{

	int8_t in_buf[13*13]={0};
	int8_t out_temp[26*13]={0};
	int8_t out_buf[26*26]={0};


	for(int cho=0;cho<Pof;cho++)
	{
		load_in_buf(In_addr ,in_buf,cho);
		horizon_upsample( in_buf , out_temp);
		vertical_upsample( out_temp, out_buf);
		load_out(Out_addr ,out_buf, cho);
	}
}

void load_in_buf(int8_t *In_addr ,int8_t *in_buf,int32_t cho)
{
	for (int i=0;i<Pir;i++)
		for(int j=0;j<Pic;j++)
		{
			in_buf[i*Pic+j]=In_addr[(cho*Pir+i)*Pic+j];
		}
}

void horizon_upsample(int8_t *in_buf ,int8_t *out_temp)
{
	for(int i=0;i<Pic*Pic;i++)
	{
		out_temp[2*i]  =in_buf[i];
		out_temp[2*i+1]=in_buf[i];
	}
}

void vertical_upsample(int8_t *out_temp,int8_t *out_buf)
{
	for(int i=0;i<2*Pic;i++)
	{
		for(int j=0;j<Pic;j++)
		{
			//[2j][i]  [2j+1][i]
			out_buf[(2*j)*2*Pic+i]=out_temp[j*2*Pic+i];
			out_buf[(2*j+1)*2*Pic+i]=out_temp[j*2*Pic+i];
		}
	}
}
void load_out(int8_t *Out_addr ,int8_t *out_buf,int32_t cho)
{
	for (int i=0;i<Pc;i++)
		for(int j=0;j<Pc;j++)
		{
			Out_addr[(cho*Pc+i)*Pc+j]=out_buf[i*Pc+j];
		}
}

/*-------tb--------*/


void data_gen(int8_t In[Pof][Pir][Pic])
{
	for(int i=0;i<Pif;i++)
		for(int j=0;j<Pir;j++)
			for(int k=0;k<Pic;k++)
			{
				In[i][j][k]=i+j+k;
			}



}




void sw_out(int8_t *Out_sw,int8_t *In)
{
	for(int r=0;r<Pic;r++)
	{
		//Column:
		for(int c=0;c<Pic;c++)
		{
		    //Out_channel:
			for(int o=0;o<Pif;o++)
			{
                
                Out_sw[(o*Pc+2*r)  *Pc+2*c]=In[(o  *Pic+r)*Pic+c];
                Out_sw[(o*Pc+2*r+1)*Pc+2*c]=In[(o  *Pic+r)*Pic+c];
                Out_sw[(o*Pc+2*r)  *Pc+2*c+1]=In[(o*Pic+r)*Pic+c];
                Out_sw[(o*Pc+2*r+1)*Pc+2*c+1]=In[(o*Pic+r)*Pic+c];
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

int8_t In[Pif][Pir][Pic];
int8_t Out_sw[Pof][Pr][Pc]={0};
int8_t Out_hw[Pof][Pr][Pc]={0};


int main() { 
	int err=0;
	int8_t *Out_p=&Out_hw[0][0][0];
	int8_t *In_p=&In[0][0][0];

	data_gen(In);
	sw_out(&Out_sw[0][0][0],In_p);
	base_upsample(In_p, &Out_hw[0][0][0]);
	printf("start\n");
	err=test_print(Out_sw,Out_hw);
	printf("finish\n");
	printf("err:%d",err);


    return 0;

}


					
#include "base_layer.h"
#include "my_timer.h"
#include "xbasic_conv.h"
#include <stdio.h>
#include "xil_printf.h"
#include "sleep.h"
#include "xil_cache.h"



#define K 3
#define Pr 208
#define Pc 208
#define Pof 32
#define Pif 3
#define S 2
#define Tk 4
#define Tc 4
#define Tp 8

XScuGic Intc;               //中断控制器驱动程序实例
XScuTimer Timer;            //定时器驱动程序实例

unsigned int count_timer=0;
XBasic_conv My_test;

void data_load_bin(int8_t *In,int8_t *W,int32_t *Bias,int32_t Param[14],int8_t *Out_sw)
{
    load_int8_data(In ,  syn_data_dir +"in_q.bin", 3*416*416);
	load_int8_data(W ,  syn_data_dir +"w_q.bin" , Pof*Pif*K*K);
	load_int32_data(Bias ,  syn_data_dir +"b_q.bin" , streamsize (4*Pof));
	int16_t m0_temp[1]={0};
    load_int16_data(m0_temp , syn_data_dir +"m_0.bin" , 2);
	int16_t s0[1]={0};
	load_int16_data(s0 , syn_data_dir +"S_relu_q.bin" , 2);

	int output_channels=32;
	int output_width=208;
	int input_width=416;
	int input_channels=3;
	int ksize=3;
	int stride=2;

    Param[0]=int32_t(m0_temp[0])	;	//M0=int16_t(Param[0]);
    Param[1]=ceil(output_channels/Tk)	;	//k_loop_max=Param[1];//kloop_max=ceil(Pof/Tk)
    Param[2]=ceil(output_width*output_width/float(Tp))	;	//p_loop_max=Param[2];//ploop_max=ceil(Pr*Pc/Tp)
    Param[3]=ceil(float(input_channels)/float(Tc))	;	//c_loop_max=Param[3];//cloop_max=ceil(float(Pif)/float(Tc))
    Param[4]=output_width	;	//out_map_height=Param[4];
    Param[5]=output_width	;	//out_map_width =Param[5];
    Param[6]=output_channels	;	//out_map_cho=Param[6];
    Param[7]=input_channels	;	//in_map_chi=Param[7];
    Param[8]=input_width	;	//in_map_height =Param[8];
    Param[9]=input_width	;	//in_map_width=Param[9];
	Param[10]=ksize	;//ksize
	Param[11]=stride	;//stride
	Param[12]=1 ;//activation_enable
	Param[13]= s0[0];

	load_int8_data(Out_sw ,  syn_data_dir +"acc_Out_q.bin" , output_width*output_width*output_channels);

}

int test_print(int8_t *Out_sw,int8_t *Out_hw)
{
	int err_sum=0;
	printf("Out_sw \n");


	for(int i=0;i<32;i++)
	{
		for(int j=0;j<208;j++)
		{
			for(int k=0;k<208;k++)
			{
				//printf("%d ",Out_sw[i][j][k]);

				if(Out_sw[(i*208+ j)*208+k]!=Out_hw[(i*208+ j)*208+k])
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

int main() {
	int status;
    xil_printf("SCU Timer Interrupt Test \r\n");

    Xil_DCacheDisable();
    Xil_ICacheDisable();
    status = timer_init(&Timer);     //定时器初始化
    if (status != XST_SUCCESS) {
        xil_printf("Timer Initial Failed\r\n");
        return XST_FAILURE;
    }
    timer_intr_init(&Intc,&Timer);   //定时器中断初始化
    XScuTimer_Start(&Timer);         //启动定时器
	SD_Init();
	/*
	int8_t In[Pif][416][416];
	int8_t W[Pof][Pif][K][K];
	int32_t Bias[Pof];
	int8_t Out_sw[Pof][Pr][Pc]={0};
	int8_t Out_hw[Pof][Pr][Pc]={0};
	int32_t Param[14];
	*/
    int8_t *in;
    in=new int8_t[3*416*416];
    int8_t *W;
    W = new int8_t [32*3*3*3];
    int32_t *Bias;
    Bias = new int32_t [32];
    int8_t *Out_sw;
    Out_sw = new int8_t [32*208*208];
    int8_t *Out_hw;
    Out_hw = new int8_t [32*208*208];
    int32_t *Param;
    Param = new int32_t[14];
    data_load_bin(in,W,Bias, Param,Out_sw);

    unsigned int time_start=0;
    unsigned int time_finish=0;


    /*---------------basic conv ip-------------------*/

    XBasic_conv_Initialize(&My_test,XPAR_BASIC_CONV_0_DEVICE_ID);

    XBasic_conv_Set_In_addr(&My_test,reinterpret_cast<uintptr_t>(in));
    XBasic_conv_Set_W_addr(&My_test,reinterpret_cast<uintptr_t>(W));
    XBasic_conv_Set_B_addr(&My_test,reinterpret_cast<uintptr_t>(Bias));
    XBasic_conv_Set_Out_addr(&My_test,reinterpret_cast<uintptr_t>(Out_hw));
    XBasic_conv_Set_Param(&My_test,reinterpret_cast<uintptr_t>(Param));

    printf("start\n\r");

    if(XBasic_conv_IsReady(&My_test))
    	XBasic_conv_Start(&My_test);
    else
    	printf("err\n\r");
    time_start = count_timer;
    std::cout << "start_time:"<< time_start <<std::endl;

    while(!XBasic_conv_IsDone(&My_test)) ;

    time_finish = count_timer;
    std::cout << "time_finish:"<< time_finish <<std::endl;

    int err=0;
    err=test_print(Out_sw,Out_hw);
    printf("err:%d",err);


    return 0;
}

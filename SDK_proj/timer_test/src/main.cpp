/*
#include "xparameters.h"				//包含器件的参数信息
#include "xscutimer.h"					//定时器中断的函数声明
#include "xscugic.h"					//包含中断的函数声明
#include "xgpiops.h"					//PS端GPIO的函数声明
#include "xil_printf.h"
#include <iostream>

#define TIMER_DEVICE_ID     XPAR_XSCUTIMER_0_DEVICE_ID   //定时器ID
#define INTC_DEVICE_ID      XPAR_SCUGIC_SINGLE_DEVICE_ID //中断ID
#define TIMER_IRPT_INTR     XPAR_SCUTIMER_INTR           //定时器中断ID


//私有定时器的时钟频率 = 111.111MHz
//0.1s,   0.1*1000_000*333 - 1 = 1fca054
//0.01s,   0.01*1000_000*333 - 1 = 32dcd4
#define TIMER_LOAD_VALUE    0x32DCD4                    //定时器装载值

XScuGic Intc;               //中断控制器驱动程序实例
XScuTimer Timer;            //定时器驱动程序实例

int s_count=0;
int count_timer=0;
int print_timer_flag=0;


//定时器初始化程序
int timer_init(XScuTimer *timer_ptr)
{
	int status;
    //私有定时器初始化
    XScuTimer_Config *timer_cfg_ptr;
    timer_cfg_ptr = XScuTimer_LookupConfig(TIMER_DEVICE_ID);
    if (NULL == timer_cfg_ptr)
        return XST_FAILURE;
    status = XScuTimer_CfgInitialize(timer_ptr, timer_cfg_ptr,timer_cfg_ptr->BaseAddr);
    if (status != XST_SUCCESS)
        return XST_FAILURE;

    XScuTimer_LoadTimer(timer_ptr, TIMER_LOAD_VALUE); // 加载计数周期
    XScuTimer_EnableAutoReload(timer_ptr);            // 设置自动装载模式

    return XST_SUCCESS;
}

//定时器中断处理程序
void timer_intr_handler(void *CallBackRef)
{
    XScuTimer *timer_ptr = (XScuTimer *) CallBackRef;
    ////

    print_timer_flag=1;

    ////
    //清除定时器中断标志
    XScuTimer_ClearInterruptStatus(timer_ptr);
}

//定时器中断初始化
void timer_intr_init(XScuGic *intc_ptr,XScuTimer *timer_ptr)
{
	//初始化中断控制器
    XScuGic_Config *intc_cfg_ptr;
    intc_cfg_ptr = XScuGic_LookupConfig(INTC_DEVICE_ID);
    XScuGic_CfgInitialize(intc_ptr, intc_cfg_ptr,intc_cfg_ptr->CpuBaseAddress);
    //设置并打开中断异常处理功能
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
            (Xil_ExceptionHandler)XScuGic_InterruptHandler, intc_ptr);
    Xil_ExceptionEnable();

    //设置定时器中断
    XScuGic_Connect(intc_ptr, TIMER_IRPT_INTR,
          (Xil_ExceptionHandler)timer_intr_handler, (void *)timer_ptr);

    XScuGic_Enable(intc_ptr, TIMER_IRPT_INTR); //使能GIC中的定时器中断
    XScuTimer_EnableInterrupt(timer_ptr);      //使能定时器中断
}
*/
//main函数
#include "my_timer.h"

XScuGic Intc;               //中断控制器驱动程序实例
XScuTimer Timer;            //定时器驱动程序实例

int s_count=0;
int count_timer=0;
int print_timer_flag=0;

int main()
{
	int status;
    xil_printf("SCU Timer Interrupt Test \r\n");


    status = timer_init(&Timer);     //定时器初始化
    if (status != XST_SUCCESS) {
        xil_printf("Timer Initial Failed\r\n");
        return XST_FAILURE;
    }
    timer_intr_init(&Intc,&Timer);   //定时器中断初始化
    XScuTimer_Start(&Timer);         //启动定时器

    while(1)
    {
    	if(print_timer_flag)
    	{
    	    if(count_timer<100)
    	    {
    	    	count_timer++;
    	    }
    	    else
    	    {
    	    	count_timer=0;
    	    	s_count++;
    	    	xil_printf("%d . %d\r\n",s_count,count_timer);
    	    }
    		print_timer_flag=0;

    	}
    }




    return 0;
}

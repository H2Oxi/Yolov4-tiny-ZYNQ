#include "my_timer.h"


extern unsigned int count_timer;
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

    count_timer++;

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

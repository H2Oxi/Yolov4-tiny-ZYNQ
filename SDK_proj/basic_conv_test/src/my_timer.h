#include "xparameters.h"				//包含器件的参数信息
#include "xscutimer.h"					//定时器中断的函数声明
#include "xscugic.h"					//包含中断的函数声明
#include "xgpiops.h"					//PS端GPIO的函数声明
#include "xil_printf.h"
#include <iostream>

#define TIMER_DEVICE_ID     XPAR_XSCUTIMER_0_DEVICE_ID   //定时器ID
#define INTC_DEVICE_ID      XPAR_SCUGIC_SINGLE_DEVICE_ID //中断ID
#define TIMER_IRPT_INTR     XPAR_SCUTIMER_INTR           //定时器中断ID


int timer_init(XScuTimer *timer_ptr);
void timer_intr_init(XScuGic *intc_ptr,XScuTimer *timer_ptr);
void timer_intr_handler(void *CallBackRef);

//私有定时器的时钟频率 = 111.111MHz
//0.1s,   0.1*1000_000*333 - 1 = 1fca054
//0.01s,   0.01*1000_000*333 - 1 = 32dcd4
#define TIMER_LOAD_VALUE    0x32DCD4                    //定时器装载值



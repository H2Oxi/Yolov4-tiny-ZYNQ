#include "base_layer.h"
#include "my_timer.h"

XScuGic Intc;               //中断控制器驱动程序实例
XScuTimer Timer;            //定时器驱动程序实例

unsigned int count_timer=0;

int main() {
	int status;
	xil_printf("Object Detection System start\r\n");
    xil_printf("SCU Timer Interrupt Test \r\n");


    status = timer_init(&Timer);     //定时器初始化
    if (status != XST_SUCCESS) {
        xil_printf("Timer Initial Failed\r\n");
        return XST_FAILURE;
    }
    timer_intr_init(&Intc,&Timer);   //定时器中断初始化
    XScuTimer_Start(&Timer);         //启动定时器
	SD_Init();
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

    unsigned int time_start=0;
    unsigned int time_finish=0;
    time_start = count_timer;
    std::cout << "start_time:"<< time_start <<std::endl;
    my_yolo_tst.forward(&in[0],Out0,Out1);
    time_finish = count_timer;

    std::cout << "finish_time:"<< time_finish <<std::endl;


    save_int8_data(Out0, syn_data_dir + "hw_cpu_yolo_out0_q8.bin" , my_yolo_tst.yolohead_p5.conv2.out_shape[0]*my_yolo_tst.yolohead_p5.conv2.out_shape[1]*my_yolo_tst.yolohead_p5.conv2.out_shape[2]);
    save_int8_data(Out1, syn_data_dir + "hw_cpu_yolo_out1_q8.bin" , my_yolo_tst.yolohead_p4.conv2.out_shape[0]*my_yolo_tst.yolohead_p4.conv2.out_shape[1]*my_yolo_tst.yolohead_p4.conv2.out_shape[2]);

    delete [] Out0;
    delete [] Out1;

    std::cout << "check SD card to check the answer!:"<< time_finish <<std::endl;
    std::cout << "finish!:"<< time_finish <<std::endl;
    return 0;
}

#include "sd_io.h"


int SD_Init()
{
    FRESULT rc;

    rc = f_mount(&fatfs, "", 0);
    if(rc){
    	xil_printf("ERROR : f_mount returned %d\r\n",rc);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}



int read_int16(string Filename, int16_t * x, int length)
{
    FIL fil;
    FRESULT rc;
    UINT br;

    char * filename=(char*)(Filename.c_str());
    rc = f_open(&fil,filename,FA_READ);
    if(rc){
    	xil_printf("ERROR : f_open returned %d\r\n",rc);
    	return XST_FAILURE;
    }
    rc = f_lseek(&fil, 0);
    if(rc){
    	xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_read(&fil, x,sizeof(int16_t) * length,&br);
    if(rc){
        xil_printf("ERROR : f_read returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_close(&fil);
    if(rc){
        xil_printf(" ERROR : f_close returned %d\r\n", rc);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}


int write_int16(string Filename, int16_t * x, int length)
{
    FIL fil;
    FRESULT rc;
    UINT bw;

    char * filename=(char*)(Filename.c_str());
    rc = f_open(&fil,filename,FA_CREATE_ALWAYS | FA_WRITE);
        if(rc){
            xil_printf("ERROR : f_open returned %d\r\n",rc);
            return XST_FAILURE;
        }
    rc = f_lseek(&fil, 0);
    if(rc){
    	xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_write(&fil,x,sizeof(int16_t) * length,&bw);
        if(rc){
            xil_printf("ERROR : f_write returned %d\r\n", rc);
            return XST_FAILURE;
        }
    rc = f_close(&fil);
        if(rc){
            xil_printf("ERROR : f_close returned %d\r\n",rc);
            return XST_FAILURE;
        }
        return XST_SUCCESS;
}


int read_int8(string Filename, int8_t * x, int length)
{
    FIL fil;
    FRESULT rc;
    UINT br;

    char * filename=(char*)(Filename.c_str());
    rc = f_open(&fil,filename,FA_READ);
    if(rc){
    	xil_printf("ERROR : f_open returned %d\r\n",rc);
    	return XST_FAILURE;
    }
    rc = f_lseek(&fil, 0);
    if(rc){
    	xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_read(&fil, x,sizeof(int8_t) * length,&br);
    if(rc){
        xil_printf("ERROR : f_read returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_close(&fil);
    if(rc){
        xil_printf(" ERROR : f_close returned %d\r\n", rc);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}


int write_int8(string Filename, int8_t * x, int length)
{
    FIL fil;
    FRESULT rc;
    UINT bw;

    char * filename=(char*)(Filename.c_str());
    rc = f_open(&fil,filename,FA_CREATE_ALWAYS | FA_WRITE);
        if(rc){
            xil_printf("ERROR : f_open returned %d\r\n",rc);
            return XST_FAILURE;
        }
    rc = f_lseek(&fil, 0);
    if(rc){
    	xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_write(&fil,x,sizeof(int8_t) * length,&bw);
        if(rc){
            xil_printf("ERROR : f_write returned %d\r\n", rc);
            return XST_FAILURE;
        }
    rc = f_close(&fil);
        if(rc){
            xil_printf("ERROR : f_close returned %d\r\n",rc);
            return XST_FAILURE;
        }
        return XST_SUCCESS;
}



int read_int32(string Filename, int32_t * x, int length)
{
    FIL fil;
    FRESULT rc;
    UINT br;

    char * filename=(char*)(Filename.c_str());
    rc = f_open(&fil,filename,FA_READ);
    if(rc){
    	xil_printf("ERROR : f_open returned %d\r\n",rc);
    	return XST_FAILURE;
    }
    rc = f_lseek(&fil, 0);
    if(rc){
    	xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_read(&fil, x,sizeof(int32_t) * length,&br);
    if(rc){
        xil_printf("ERROR : f_read returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_close(&fil);
    if(rc){
        xil_printf(" ERROR : f_close returned %d\r\n", rc);
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}


int write_int32(string Filename, int32_t * x, int length)
{
    FIL fil;
    FRESULT rc;
    UINT bw;

    char * filename=(char*)(Filename.c_str());
    rc = f_open(&fil,filename,FA_CREATE_ALWAYS | FA_WRITE);
        if(rc){
            xil_printf("ERROR : f_open returned %d\r\n",rc);
            return XST_FAILURE;
        }
    rc = f_lseek(&fil, 0);
    if(rc){
    	xil_printf("ERROR : f_lseek returned %d\r\n",rc);
        return XST_FAILURE;
    }
    rc = f_write(&fil,x,sizeof(int32_t) * length,&bw);
        if(rc){
            xil_printf("ERROR : f_write returned %d\r\n", rc);
            return XST_FAILURE;
        }
    rc = f_close(&fil);
        if(rc){
            xil_printf("ERROR : f_close returned %d\r\n",rc);
            return XST_FAILURE;
        }
        return XST_SUCCESS;
}


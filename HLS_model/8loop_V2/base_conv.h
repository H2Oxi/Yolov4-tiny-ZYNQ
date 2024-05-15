#ifndef __CONV_H__
#define __CONV_H__



#define K 3
#define Pr 16
#define Pc 16
#define Pof 16
#define Pif 3
#define S 2

#define Tk 4
#define Tc 4
#define Tp 8

//kloop_max=ceil(Pof/Tk)
#define K_LOOP_MAX 4
//ploop_max=ceil(Pr*Pc/Tp)
#define P_LOOP_MAX 32
//cloop_max=ceil(float(Pif)/float(Tc))
#define C_LOOP_MAX 1



void base_conv(int8_t *In_addr,int8_t *W_addr,int32_t *B_addr,int8_t *Out_addr,int32_t *Param);


#endif

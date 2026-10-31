/* Numerical contract: a constant silent spectrogram normalizes to zero. */
#include "nd_whistle.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
int main(void) {
    nd_whistle w={0};nd_tensor t[2]={{0}};const void *data[2];
    float manifest[13]={3246,2,2048,8,4,9,128,3,1,16000,512,400,160};
    static float fb[257],pcm[16000],out[100];
    t[0].dtype=ND_DT_FP32;data[0]=manifest;data[1]=fb;
    w.t=t;w.p=data;w.i_audio_cfg=0;w.i_melfb=1;w.mels=1;w.n_fft=512;w.win=400;w.hop=160;
    /* Zero input and zero mel energy require no nonzero FFT/window coefficients. */
    nd_whistle_logmel(&w,pcm,16000,out);
    for(unsigned i=0;i<100;i++) {
        if(!isfinite(out[i]) || fabsf(out[i])>1e-6f) {
            fprintf(stderr,"silent profile-2 feature[%u] = %.9g, expected zero\n",i,out[i]);return 1;
        }
    }
    puts("Research frontend: silent features remain finite and zero");
}

/* Compare gate boundaries against the former two-buffer implementation and
 * measure real malloc traffic, including failure of the percentile buffer. */
#include "nd_whistle.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__real_malloc(size_t size);
void __real_free(void *pointer);
static unsigned calls, frees;
static size_t requested;
static int fail_allocation;
void *__wrap_malloc(size_t size) {
    calls++;
    requested += size;
    return fail_allocation ? NULL : __real_malloc(size);
}
void __wrap_free(void *pointer) {
    if (pointer) frees++;
    __real_free(pointer);
}

/* The pre-change gate, with stack buffers solely for the reference result.
 * Its energy summation order and percentile rule are intentionally retained. */
static int reference(const float *x, uint32_t n, float floor_rms,
                     float db, uint32_t margin, uint32_t *s0, uint32_t *s1) {
    float rms[400], sorted[400], threshold;
    uint32_t windows=n/320, loud=0, first=0, last=0;
    assert(windows<=400);
    if(windows<5)return 0;
    for(uint32_t i=0;i<windows;i++) {
        float energy=0;
        for(uint32_t k=0;k<320;k++)energy+=x[i*320+k]*x[i*320+k];
        rms[i]=sqrtf(energy/320.0f);
    }
    if(floor_rms<=0) {
        memcpy(sorted,rms,windows*sizeof(float));
        for(uint32_t a=1;a<windows;a++)
            for(uint32_t b=a;b>0 && sorted[b-1]>sorted[b];b--) {
                float t=sorted[b];sorted[b]=sorted[b-1];sorted[b-1]=t;
            }
        floor_rms=sorted[windows/10];
    }
    threshold=(floor_rms>1e-7f?floor_rms:1e-7f)*powf(10.0f,db/20.0f);
    for(uint32_t i=0;i<windows;i++)if(rms[i]>threshold) {
        if(!loud)first=i;
        last=i;loud++;
    }
    if(loud<5)return 0;
    *s0=first*320>margin*16?first*320-margin*16:0;
    *s1=(last+1)*320+margin*16<n?(last+1)*320+margin*16:n;
    return 1;
}

static float audio[128000];
static void check(uint32_t n, float floor_rms, float db, uint32_t margin) {
    uint32_t expected_start=77,expected_end=88,start=77,end=88;
    int expected=reference(audio,n,floor_rms,db,margin,&expected_start,&expected_end);
    calls=frees=0;requested=0;
    int actual=nd_whistle_find_speech(audio,n,floor_rms,db,margin,&start,&end);
    assert(actual==expected && start==expected_start && end==expected_end);
    unsigned allocations=n/320>=5 && floor_rms<=0 ? 1 : 0;
    assert(calls==allocations && frees==allocations);
    assert(requested==(allocations?n/320*sizeof(float):0));
}

int main(void) {
    uint32_t seed=0x51ee73;
    unsigned comparisons=0;
    /* Silence, boundary clips, 8-second maximum, constant/noisy audio and
     * intermittent voiced windows; both automatic and explicit floors. */
    for(unsigned trial=0;trial<96;trial++) {
        uint32_t n=trial==0?128000:trial==1?1599:trial==2?1600:
                     320*(5+trial*37%395)+(trial%17);
        for(uint32_t i=0;i<n;i++) {
            seed=seed*1664525u+1013904223u;
            float gain=(i/320)%13<4?0.00001f:0.1f;
            audio[i]=trial==0?0:trial==3?0.03f:
                       ((float)(seed>>16)/32768.0f-1.0f)*gain;
        }
        for(unsigned floor=0;floor<3;floor++)
            for(unsigned gate=0;gate<3;gate++)
                for(unsigned margin=0;margin<2;margin++) {
                    check(n,floor==0?0:floor==1?0.001f:0.01f,
                          6.0f+gate*6.0f,margin*300);
                    comparisons++;
                }
    }
    fail_allocation=1;calls=frees=0;requested=0;
    uint32_t start=77,end=88;
    assert(!nd_whistle_find_speech(audio,128000,0,12,300,&start,&end));
    assert(calls==1 && frees==0 && requested==1600 && start==77 && end==88);
    check(128000,0.001f,12,300); /* An explicit floor still needs no heap. */
    puts("PASS speech gate: 1728 identical boundary comparisons; automatic peak 1600 vs 3200 bytes; fixed floor zero heap; allocation failure safe");
    assert(comparisons==1728);
}

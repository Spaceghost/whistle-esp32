#include "nd_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
static unsigned internal_calls,psram_calls;
static size_t internal_limit,psram_limit;
void *heap_caps_malloc(size_t n,unsigned caps) {
    assert(caps&MALLOC_CAP_8BIT);
    if(caps&MALLOC_CAP_INTERNAL){internal_calls++;return n<=internal_limit?malloc(n):NULL;}
    assert(caps&MALLOC_CAP_SPIRAM);psram_calls++;return n<=psram_limit?malloc(n):NULL;
}
int main(void) {
    /* Reproduce a fragmented internal heap with enough total PSRAM for the
     * 32 KB GEMM tile that previously reached memcpy(NULL,...). */
    internal_limit=24000;psram_limit=8*1024*1024;
    unsigned char *p=nd_alloc_fast(33792);assert(p);memset(p,0x5a,33792);
    assert(p[33791]==0x5a && internal_calls==1 && psram_calls==1);free(p);
    internal_calls=psram_calls=0;internal_limit=65536;
    p=nd_alloc_fast(1024);assert(p && internal_calls==1 && psram_calls==0);free(p);
    internal_limit=psram_limit=0;assert(nd_alloc_fast(1)==NULL);
    puts("PASS fast scratch prefers SRAM and falls back on fragmented SRAM");
}

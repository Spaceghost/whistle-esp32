#ifndef ND_ALLOC_H
#define ND_ALLOC_H
#include <stddef.h>
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
/* Hot scratch prefers internal SRAM, but a fragmented TLS/UI heap must not
 * turn a valid inference into a NULL write. Scratch is not DMA/ISR memory. */
static inline void *nd_alloc_fast(size_t bytes)
{
    void *p = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    return p ? p : heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
}
#else
#include <stdlib.h>
static inline void *nd_alloc_fast(size_t bytes) { return malloc(bytes); }
#endif
#endif

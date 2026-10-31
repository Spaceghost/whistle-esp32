/* Host-only requested-heap accounting over the unchanged Whistle engine.
 * The model is mmap'd, as it is on the Watch. Tracker/libc bookkeeping,
 * allocator overhead, stack, task heaps and physical fragmentation are excluded.
 * This measures the current malloc/calloc/free engine path, not board acceptance.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "nd_whistle.h"

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void __real_free(void *);
static struct { void *pointer; size_t bytes; } tracked[4096];
static size_t live, peak, calls, failures;
static int measuring;

static void record(void *p, size_t bytes) {
    if (!measuring) return;
    calls++;
    if (!p) { failures++; return; }
    for (unsigned i = 0; i < 4096; i++) if (!tracked[i].pointer) {
        tracked[i].pointer = p; tracked[i].bytes = bytes; live += bytes;
        if (live > peak) peak = live;
        return;
    }
    abort();
}
void *__wrap_malloc(size_t bytes) {
    void *p = __real_malloc(bytes); record(p, bytes); return p;
}
void *__wrap_calloc(size_t n, size_t bytes) {
    void *p = __real_calloc(n, bytes);
    if (p && bytes && n > SIZE_MAX / bytes) abort();
    record(p, n * bytes); return p;
}
void __wrap_free(void *p) {
    if (p) for (unsigned i = 0; i < 4096; i++) if (tracked[i].pointer == p) {
        live -= tracked[i].bytes; tracked[i].pointer = NULL; break;
    }
    __real_free(p);
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char *end = NULL;
    errno = 0;
    unsigned long seconds = strtoul(argv[2], &end, 10);
    if (errno || !*argv[2] || *end || seconds < 1 || seconds > 12) return 2;
    int fd = open(argv[1], O_RDONLY);
    struct stat info;
    if (fd < 0) return 3;
    if (fstat(fd, &info) || info.st_size <= 0 || info.st_size > 64*1024*1024) { close(fd); return 3; }
    void *blob = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (blob == MAP_FAILED) return 3;
    size_t samples = seconds * 16000;
    /* Caller PCM is outside the engine heap counter; account separately. */
    float *pcm = __real_calloc(samples, sizeof(float));
    if (!pcm) { munmap(blob, (size_t)info.st_size); return 3; }
    nd_whistle w = {0};
    int rc = 1, tokens = -1;
    float *enc = NULL;
    size_t opened = 0, open_peak = 0, encoded = 0, encode_peak = 0, decode_peak = 0;
    measuring = 1;
    if (nd_whistle_open(&w, blob, (size_t)info.st_size)) goto done;
    opened = live; open_peak = peak;
    unsigned frames = nd_whistle_frames(&w, (uint32_t)samples);
    encoded = (size_t)frames * w.d * sizeof(float);
    enc = __wrap_malloc(encoded);
    if (!enc) goto done;
    peak = live;
    if (nd_whistle_encode(&w, pcm, (uint32_t)samples, enc)) goto done;
    encode_peak = peak;
    if (live != opened + encoded) goto done;
    peak = live;
    uint32_t ids[128];
    tokens = nd_whistle_decode(&w, enc, frames, 0, ids, 128);
    decode_peak = peak;
    if (tokens < 0 || live != opened + encoded) goto done;
    rc = 0;
done:
    nd_whistle_close(&w);
    __wrap_free(enc);
    measuring = 0;
    __real_free(pcm);
    munmap(blob, (size_t)info.st_size);
    if (live || failures) rc = 1;
    printf("{\"seconds\":%lu,\"model_state_host_bytes\":%zu,\"model_open_live_bytes\":%zu,"
           "\"model_open_peak_bytes\":%zu,\"encoded_output_bytes\":%zu,\"encode_peak_bytes\":%zu,"
           "\"decode_peak_bytes\":%zu,\"one_pcm_buffer_bytes\":%zu,\"final_live_bytes\":%zu,"
           "\"allocation_calls\":%zu,\"allocation_failures\":%zu,\"decode_tokens\":%d,\"passed\":%s}\n",
           seconds, sizeof(w), opened, open_peak, encoded, encode_peak, decode_peak,
           samples*sizeof(float), live, calls, failures, tokens, rc ? "false" : "true");
    return rc;
}

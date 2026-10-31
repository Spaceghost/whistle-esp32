/* Host-only decoder boundary access. Production code and APIs are unchanged. */
#include <stdio.h>
#include "../engine/src/nd_whistle.c"

static void *load(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb"); void *p = NULL;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) || ftell(f) < 0) goto done;
    *size = (size_t)ftell(f);
    if (!*size || *size > 64u*1024u*1024u || fseek(f, 0, SEEK_SET)) goto done;
    p = malloc(*size);
    if (p && fread(p, 1, *size, f) != *size) { free(p); p = NULL; }
done: fclose(f); return p;
}
static int save(const char *root, const char *name, const float *data, size_t count) {
    char path[4096]; int n = snprintf(path, sizeof(path), "%s/%s.f32", root, name);
    if (n < 0 || (size_t)n >= sizeof(path)) return 1;
    FILE *f = fopen(path, "wbx"); if (!f) return 1;
    int rc = fwrite(data, sizeof(float), count, f) != count;
    return fclose(f) != 0 || rc;
}
int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) return 2;
    if (argc == 6) { if (strcmp(argv[5], "--float")) return 2; nd_cq_int8 = 0; }
    size_t bytes = 0, enc_bytes = 0, tok_bytes = 0;
    void *blob = load(argv[1], &bytes);
    float *enc = load(argv[2], &enc_bytes), *hidden = NULL, *logits = NULL;
    uint32_t *tokens = load(argv[3], &tok_bytes);
    nd_whistle w = {0}; int rc = 1, session = 0;
    if (!blob || !enc || !tokens || nd_whistle_open(&w, blob, bytes)) goto done;
    size_t row = (size_t)w.d * sizeof(float);
    if (!row || enc_bytes % row || enc_bytes / row > 400 || tok_bytes % sizeof(uint32_t)) goto done;
    unsigned T = (unsigned)(enc_bytes / row), n = (unsigned)(tok_bytes / sizeof(uint32_t));
    if (!T || !n || n > w.max_seq) goto done;
    for (size_t i = 0; i < enc_bytes / sizeof(float); i++) if (!isfinite(enc[i])) goto done;
    for (unsigned i = 0; i < n; i++) if (tokens[i] >= w.vocab) goto done;
    hidden = malloc((size_t)n*w.d*sizeof(float));
    logits = malloc((size_t)n*w.vocab*sizeof(float));
    if (!hidden || !logits || session_begin(&w, enc, T)) goto done;
    session = 1;
    if (save(argv[4], "cross-k", w.cross_k, (size_t)w.dec_layers*w.nh*T*w.qk) ||
        save(argv[4], "cross-v", w.cross_v, (size_t)w.dec_layers*w.nh*T*w.vd)) goto done;
    for (unsigned i = 0; i < n; i++) {
        dec_step(&w, tokens[i], i, hidden+(size_t)i*w.d);
        logits_all(&w, hidden+(size_t)i*w.d, logits+(size_t)i*w.vocab);
    }
    rc = save(argv[4], "hidden", hidden, (size_t)n*w.d) ||
         save(argv[4], "logits", logits, (size_t)n*w.vocab);
done:
    if (session) session_end(&w);
    nd_whistle_close(&w); free(blob); free(enc); free(tokens); free(hidden); free(logits);
    return rc;
}

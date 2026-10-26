/* Exercise the production tokenizer with Whistle-sized metadata, collisions,
 * duplicate surfaces, byte fallback, marker prefixes and allocation failures. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nd_cact.h"

static size_t allocated, calls, fail_at;
static void *counted_calloc(size_t count, size_t size)
{
    calls++;
    if (fail_at && calls == fail_at) return NULL;
    allocated += count * size;
    return calloc(count, size);
}
#define calloc counted_calloc
#include "../engine/src/nd_tokenizer.c"
#undef calloc

static void put32(uint8_t *p, uint32_t n)
{
    p[0] = n; p[1] = n >> 8; p[2] = n >> 16; p[3] = n >> 24;
}

static size_t make_blob(uint8_t *blob, uint32_t count, uint32_t markers)
{
    size_t off = 24;
    memset(blob, 0, 24);
    put32(blob, count);
    blob[21] = 1;
    for (uint32_t i = 0; i < count; i++) {
        char text[40];
        uint8_t type = ND_TK_NORMAL;
        if (i < 256) {
            snprintf(text, sizeof(text), "<0x%02X>", i);
            type = ND_TK_BYTE;
        } else if (i < 256 + markers) {
            /* Prefix-related markers must remain atomic longest-first. */
            memset(text, 'm', i - 255);
            text[i - 255] = 0;
            type = ND_TK_USER_DEFINED;
        } else if (i == count - 1) {
            strcpy(text, "<0x41>"); /* duplicate must resolve to lower ID */
        } else {
            snprintf(text, sizeof(text), "piece-%05u", i);
        }
        size_t len = strlen(text);
        put32(blob + off, 0);
        blob[off + 4] = type;
        blob[off + 5] = len; blob[off + 6] = len >> 8;
        memcpy(blob + off + 7, text, len);
        off += 7 + len;
    }
    return off;
}

static void compare_legacy(nd_tokenizer *small)
{
    nd_tokenizer legacy = *small;
    uint32_t buckets = 1;
    while (buckets < legacy.n_pieces * 2) buckets <<= 1;
    legacy.mask = buckets - 1;
    legacy.bucket = calloc(buckets, sizeof(*legacy.bucket));
    assert(legacy.bucket);
    for (uint32_t i = 0; i < legacy.n_pieces; i++) {
        uint16_t len;
        const char *s = surface(&legacy, i, &len);
        if (lookup(&legacy, s, len) < 0) insert(&legacy, i);
        assert(lookup(small, s, len) == lookup(&legacy, s, len));
    }
    assert(lookup(small, "absent", 6) == -1);
    const char *texts[] = {"Hello Levi. This is a connection test.",
        "mmmmm hello mmmmmmmmmmm world", "café — hello", "", "piece-03000"};
    for (size_t i = 0; i < sizeof(texts) / sizeof(texts[0]); i++) {
        for (int dummy = 0; dummy <= 1; dummy++) {
            uint32_t a[512], b[512];
            char da[2048], db[2048];
            int na = nd_tok_encode_ex(small, texts[i], strlen(texts[i]), a, 512, dummy);
            int nb = nd_tok_encode_ex(&legacy, texts[i], strlen(texts[i]), b, 512, dummy);
            assert(na >= 0 && na == nb);
            assert(memcmp(a, b, (size_t)na * sizeof(*a)) == 0);
            assert(nd_tok_decode(small, a, na, da, sizeof(da)) ==
                   nd_tok_decode(&legacy, b, nb, db, sizeof(db)));
            assert(strcmp(da, db) == 0);
        }
    }
    free(legacy.bucket);
}

static void check_model(const char *path)
{
    FILE *f = fopen(path, "rb");
    assert(f && fseek(f, 0, SEEK_END) == 0);
    long len = ftell(f);
    assert(len > 0 && fseek(f, 0, SEEK_SET) == 0);
    void *data = malloc((size_t)len);
    assert(data && fread(data, 1, (size_t)len, f) == (size_t)len);
    fclose(f);
    nd_cact model;
    assert(nd_cact_open(&model, data, (size_t)len) == 0);
    nd_tensor tensor;
    int found = 0;
    for (uint32_t i = model.n; i-- > 0;) {
        assert(nd_cact_tensor(&model, i, &tensor) == 0);
        if (tensor.dtype == ND_DT_RAW) { found = 1; break; }
    }
    assert(found);
    nd_tokenizer t;
    allocated = calls = 0;
    assert(nd_tok_init(&t, nd_cact_data(&model, &tensor), tensor.nbytes) == 0);
    size_t before = 1;
    while (before < t.n_pieces * 2) before <<= 1;
    size_t saved = (before - (t.mask + 1)) * sizeof(uint16_t) +
                   (t.n_pieces - t.n_markers) * sizeof(uint32_t);
    compare_legacy(&t);
    printf("PASS actual model: %u pieces, %u markers; %zu metadata bytes saved; all lookups and corpus identical\n",
           t.n_pieces, t.n_markers, saved);
    nd_tok_free(&t);
    free(data);
}

int main(int argc, char **argv)
{
    uint8_t *blob = malloc(8199 * 48 + 24);
    assert(blob);
    size_t size = make_blob(blob, 8199, 31);
    nd_tokenizer t;
    allocated = calls = 0;
    assert(nd_tok_init(&t, blob, size) == 0);
    assert(t.n_markers == 31 && t.mask + 1 == 16384);
    size_t saved = (32768 - 16384) * sizeof(uint16_t) + (8199 - 31) * sizeof(uint32_t);
    assert(saved == 65440);
    assert(allocated == 8199 * sizeof(nd_piece) + 16384 * sizeof(uint16_t) + 31 * sizeof(uint32_t));
    for (uint32_t i = 1; i < t.n_markers; i++)
        assert(t.pieces[t.markers[i - 1]].len >= t.pieces[t.markers[i]].len);
    assert(lookup(&t, "<0x41>", 6) == 65);
    compare_legacy(&t);
    nd_tok_free(&t);
    for (fail_at = 1; fail_at <= 3; fail_at++) {
        calls = 0;
        assert(nd_tok_init(&t, blob, size) < 0);
        nd_tok_free(&t);
    }
    fail_at = 0;
    size = make_blob(blob, 8199, 0);
    calls = allocated = 0;
    assert(nd_tok_init(&t, blob, size) == 0);
    assert(t.markers == NULL && t.n_markers == 0 && calls == 2);
    compare_legacy(&t);
    nd_tok_free(&t);
    free(blob);
    if (argc == 2) check_model(argv[1]);
    printf("PASS tokenizer: %zu metadata bytes saved; lookup/encoding parity; allocation failures; zero markers\n", saved);
}

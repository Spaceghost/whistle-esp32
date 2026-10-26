#include "nd_whistle.h"
#include "nd_ledcmd.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    nd_whistle w = {0};
    uint32_t ids[8];
    assert(ND_WH_LANGS == 1);
    /* Invalid language requests must fail before dereferencing a model. */
    assert(nd_whistle_decode(&w, NULL, 0, 1, ids, 8) == -1);
    assert(nd_whistle_decode(&w, NULL, 0, -2, ids, 8) == -1);
    assert(nd_led_keywords(0));
    for (int i = 1; i < 7; ++i) assert(nd_led_keywords(i) == NULL);
    printf("PASS English-only API; nd_whistle state=%zu bytes; keyword capacity=%d\n",
           sizeof(w), ND_WH_MAX_KW);
}

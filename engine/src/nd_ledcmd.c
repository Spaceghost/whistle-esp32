#include "nd_ledcmd.h"

#include <stdlib.h>
#include <string.h>

/* word -> canonical colour. '*' at the end matches as a stem. Short words
 * stay exact so they do not catch others ("rot" vs "rotate", "verde" vs
 * Spanish "verdad"). */
static const struct { const char *w, *color; } COLOURS[] = {
    /* en */ { "red", "red" }, { "read", "red" }, { "green", "green" }, { "blue", "blue" },
             { "blew", "blue" }, { "yellow", "yellow" }, { "purple", "purple" }, { "white", "white" },

};

/* English command vocabulary only. */
static const char *const FLASH[] = { "flash", "blink", "strob", "flicker", "puls" };
static const char *const SECOND[] = { "second" };
static const char *const EVERY[] = { "every" };
static const struct { const char *w; float v; } NUMBERS[] = {
    { "a", 1 }, { "one", 1 }, { "two", 2 }, { "three", 3 }, { "four", 4 }, { "five", 5 },
    { "six", 6 }, { "seven", 7 }, { "eight", 8 }, { "nine", 9 }, { "ten", 10 }, { "half", 0.5f },
};
static const char *const KEYWORDS[] = {
    "red\ngreen\nblue\nyellow\npurple\nwhite\nflash\nblink",
};

const char *nd_led_keywords(int lang)
{
    return lang >= 0 && lang < (int)(sizeof(KEYWORDS) / sizeof(KEYWORDS[0])) ? KEYWORDS[lang] : NULL;
}

void nd_utf8_lower(char *s)
{
    /* Preserve non-ASCII bytes so foreign words cannot become English words. */
    for (; *s; ++s)
        if (*s >= 'A' && *s <= 'Z')
            *s += 'a' - 'A';
}

static int is_letter(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80;
}

/* Next word, lowercased; "0,5" and "0.5" stay one word (as "0.5"). */
static int next_word(const char **p, char *w, int cap)
{
    const unsigned char *s = (const unsigned char *)*p;
    int                  n = 0;
    while (*s && !is_letter(*s))
        s++;
    while (*s && (is_letter(*s) ||
                  ((*s == '.' || *s == ',') && n && w[n - 1] >= '0' && w[n - 1] <= '9' &&
                   s[1] >= '0' && s[1] <= '9'))) {
        if (n < cap - 1)
            w[n++] = (char)(*s == ',' ? '.' : *s);
        s++;
    }
    w[n] = '\0';
    *p = (const char *)s;
    nd_utf8_lower(w);
    return n;
}

static int match(const char *word, const char *pat)
{
    size_t n = strlen(pat);
    if (n && pat[n - 1] == '*')
        return !strncmp(word, pat, n - 1);
    return !strcmp(word, pat);
}

static int in_list(const char *w, const char *const *list, int n, int stem)
{
    int i;
    for (i = 0; i < n; i++)
        if (stem ? !strncmp(w, list[i], strlen(list[i])) : !strcmp(w, list[i]))
            return 1;
    return 0;
}

static float number(const char *w)
{
    int i;
    if (w[0] >= '0' && w[0] <= '9')
        return (float)atof(w);
    for (i = 0; i < (int)(sizeof(NUMBERS) / sizeof(NUMBERS[0])); i++)
        if (!strcmp(w, NUMBERS[i].w))
            return NUMBERS[i].v;
    return -1.0f;
}

#define N(a) ((int)(sizeof(a) / sizeof((a)[0])))

int nd_led_parse(const char *text, const char *const *colors, int n_colors,
                 float default_s, nd_led_cmd *out)
{
    const char *p = text;
    char        w[40], h[3][40] = { "", "", "" };   /* h[0] = previous word */
    int         i, k;

    out->color   = -1;
    out->flash   = 0;
    out->seconds = default_s;
    out->period  = 0.0f;
    while (next_word(&p, w, sizeof(w))) {
        for (i = 0; i < N(COLOURS); i++)
            if (match(w, COLOURS[i].w))
                for (k = 0; k < n_colors; k++)
                    if (!strcmp(COLOURS[i].color, colors[k]))
                        out->color = k;
        if (in_list(w, FLASH, N(FLASH), 1))
            out->flash = 1;
        if (in_list(w, SECOND, N(SECOND), 1)) {
            int   every = 0;
            float v = number(h[0]);
            if (!strcmp(h[0], "a") && !strcmp(h[1], "half"))
                v = 0.5f;                              /* "half a second" */
            for (k = 0; k < 3; k++)
                every |= in_list(h[k], EVERY, N(EVERY), 0);
            if (every) {
                out->period = v > 0.0f && v <= 10.0f ? v : 1.0f;   /* "every second" */
                out->flash  = 1;
            } else if (v > 0.0f && v <= 30.0f) {     /* led.json: maximum 30 */
                out->seconds = v;
            }
        }
        memcpy(h[2], h[1], sizeof(h[2]));
        memcpy(h[1], h[0], sizeof(h[1]));
        memcpy(h[0], w, sizeof(h[0]));
    }
    return out->color >= 0;
}

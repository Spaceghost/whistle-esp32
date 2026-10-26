/* nd_ledcmd.h - turn a Whistle transcript into an LED command.
 *
 * Whistle gives text, not a tool call, so the LED demo maps the transcript
 * onto the set_led tool's arguments (tools/led.json) directly: a colour from
 * the caller's (English) list, flash or solid, a duration and a blink period.
 *
 * English colour and blink words only. The last named colour wins.
 */
#ifndef ND_LEDCMD_H
#define ND_LEDCMD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   color;      /* index into the caller's colour list, or -1 */
    int   flash;      /* 1 for flash / blink, 0 for solid */
    float seconds;    /* stated duration, or the default */
    float period;     /* "every N seconds": one on/off cycle, or 0 */
} nd_led_cmd;

/* Returns 1 if a colour was found. `colors` are English names (red, green,
 * blue, yellow, purple, white); `seconds` defaults to default_s. */
int nd_led_parse(const char *text, const char *const *colors, int n_colors,
                 float default_s, nd_led_cmd *out);

/* English keywords for lang 0; NULL for every other index. */
const char *nd_led_keywords(int lang);

/* Lowercase ASCII in place, preserving other UTF-8 bytes. */
void nd_utf8_lower(char *s);

#ifdef __cplusplus
}
#endif
#endif /* ND_LEDCMD_H */

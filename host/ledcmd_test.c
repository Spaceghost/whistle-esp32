/* ledcmd_test - transcripts Whistle produced on the board, and what the LED
 * demo must make of them. */
#include <stdio.h>
#include <string.h>

#include "nd_ledcmd.h"

static const char *const COLORS[] = { "red", "green", "blue", "yellow", "purple", "white" };

static const struct { const char *text; int color, flash; float secs, period; } CASES[] = {
    { "flash delight red.",                         0, 1, 3.0f, 0.0f },
    { "Meg delight yellow for five seconds.",       3, 0, 5.0f, 0.0f },
    { "Make delight purple for two seconds.",       4, 0, 2.0f, 0.0f },
    { "Strobed the Blue Light every 0.5 seconds.",  2, 1, 3.0f, 0.5f },
    { "Make the light yellow for nine seconds.",    3, 0, 9.0f, 0.0f },
    { "Flash read for two seconds.",                0, 1, 2.0f, 0.0f },
    { "Blinking green, every second.",              1, 1, 3.0f, 1.0f },
    { "What is the capital of France?",            -1, 0, 3.0f, 0.0f },
    { "Turn it white for 3 seconds",                5, 0, 3.0f, 0.0f },
    /* Removed command languages must no longer dispatch an LED action. */
    { "Fais clignoter la lumière rouge.",          -1, 0, 3.0f, 0.0f },
    { "Lass das Licht rot blinken.",              -1, 0, 3.0f, 0.0f },
    { "Pon la luz amarilla.",                     -1, 0, 3.0f, 0.0f },
    { "Fai lampeggiare la luce viola.",           -1, 0, 3.0f, 0.0f },
    { "Laat het licht groen knipperen.",          -1, 0, 3.0f, 0.0f },
    { "Niech światło miga na czerwono.",          -1, 0, 3.0f, 0.0f },
    /* The README's examples */
    { "Make it red.",                               0, 0, 3.0f, 0.0f },
    { "Turn the light green for five seconds.",     1, 0, 5.0f, 0.0f },
    { "Flash the light blue.",                      2, 1, 3.0f, 0.0f },
    { "Blink purple every half a second.",          4, 1, 3.0f, 0.5f },

};

int main(int argc, char **argv)
{
    int i, fails = 0;
    if (argc > 1 && !strcmp(argv[1], "--stdin")) {   /* parse transcripts, one per line */
        char line[1024];
        while (fgets(line, sizeof(line), stdin)) {
            nd_led_cmd c;
            if (nd_led_parse(line, COLORS, 6, 3.0f, &c))
                printf("ACT led color=%s mode=%s\n", COLORS[c.color], c.flash ? "flash" : "solid");
            else
                printf("ACT none\n");
        }
        return 0;
    }
    for (i = 0; i < (int)(sizeof(CASES) / sizeof(CASES[0])); i++) {
        nd_led_cmd c;
        int        found = nd_led_parse(CASES[i].text, COLORS, 6, 3.0f, &c);
        int        ok = c.color == CASES[i].color && (!found || (c.flash == CASES[i].flash &&
                        c.seconds == CASES[i].secs && c.period == CASES[i].period));
        printf("%s  %-44s -> %s %s %.1fs period %.1f\n", ok ? "ok  " : "FAIL", CASES[i].text,
               found ? COLORS[c.color] : "none", c.flash ? "flash" : "solid", c.seconds, c.period);
        fails += !ok;
    }
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}

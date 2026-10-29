/* Compare sparse undo against the previous full-copy keyword scorer.
 * Includes repeated targets, prefixes, already-said groups, ties, tiny bonuses
 * lost to rounding, negative bonuses and the complete keyword capacity. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../engine/src/nd_whistle.c"

static void reference_bias_keywords(const nd_whistle *w, const uint32_t *ids, uint32_t n,
                          float *lg)
{
    uint8_t  done[ND_WH_MAX_KW];      /* phrase group already said */
    uint32_t q, k, j;
    char     text[512];

    if (!w->n_kw)
        return;
    if (nd_tok_decode(&w->tok, ids, n, text, sizeof(text)) < 0)
        text[0] = '\0';
    memset(done, 0, sizeof(done));
    for (q = 0; q < w->n_kw; q++)
        if (said(text, w->kw_text[w->kw_group[q]]))
            done[w->kw_group[q]] = 1;
    for (q = 0; q < w->n_kw; q++) {
        uint32_t m = w->kw_len[q], best = 0;
        if (done[w->kw_group[q]])
            continue;
        for (k = (m - 1 < n ? m - 1 : n); k > 0; k--) {
            for (j = 0; j < k && ids[n - k + j] == w->kw[q][j]; j++)
                ;
            if (j == k) {
                best = k;
                break;
            }
        }
        lg[w->kw[q][best]] += w->kw_bonus;
    }
}


static unsigned rng = 0x921ea7u;
static unsigned next_random(void) { rng = rng * 1664525u + 1013904223u; return rng; }
static void put32(unsigned char *p, unsigned x) {
    p[0]=x;p[1]=x>>8;p[2]=x>>16;p[3]=x>>24;
}
int main(void) {
    enum { VOCAB=8193 };
    static nd_whistle w;
    static float original[VOCAB], expected[VOCAB], actual[VOCAB];
    const char *pieces[]={"<pad>","</s>","<s>","<unk>",
        "\xe2\x96\x81" "hello","\xe2\x96\x81" "levi","\xe2\x96\x81" "attack",
        "\xe2\x96\x81" "on","\xe2\x96\x81" "titan","\xe2\x96\x81" "app",
        "\xe2\x96\x81" "caf\xc3\xa9","\xe2\x96\x81" "leviathan"};
    unsigned char blob[512]={0};size_t offset=24;
    put32(blob,sizeof(pieces)/sizeof(*pieces));put32(blob+4,0);put32(blob+8,1);put32(blob+12,2);put32(blob+16,3);blob[20]=1;
    for(unsigned i=0;i<sizeof(pieces)/sizeof(*pieces);i++) {
        size_t len=strlen(pieces[i]);blob[offset+4]=i<4?ND_TK_CONTROL:ND_TK_NORMAL;
        blob[offset+5]=len;memcpy(blob+offset+7,pieces[i],len);offset+=7+len;
    }
    assert(nd_tok_init(&w.tok,blob,offset)==0);w.vocab=VOCAB;
    const char *groups[]={"levi","attack on titan","app","café","hello","never said","titan","on"};
    for(unsigned g=0;g<8;g++)strcpy(w.kw_text[g],groups[g]);
    /* Deterministic boundary witnesses independent of randomized coverage. */
    keyword_undo edge[ND_WH_MAX_KW];uint32_t prefix[1]={6};
    w.n_kw=1;w.kw_bonus=4.0f;w.kw_group[0]=0;w.kw_len[0]=1;
    strcpy(w.kw_text[0],"never said");w.kw[0][0]=EOS_ID;
    assert(bias_keywords(&w,NULL,0,actual,edge)==1 && actual[EOS_ID]==4.0f);
    actual[edge[0].token]=edge[0].original;
    w.n_kw=2;w.kw_group[1]=0;w.kw_len[1]=1;w.kw[1][0]=EOS_ID;
    assert(bias_keywords(&w,NULL,0,actual,edge)==1 && actual[EOS_ID]==8.0f);
    actual[edge[0].token]=edge[0].original;
    w.n_kw=1;w.kw_len[0]=2;w.kw[0][0]=6;w.kw[0][1]=7;
    assert(bias_keywords(&w,prefix,1,actual,edge)==1 && actual[7]==4.0f);
    actual[edge[0].token]=edge[0].original;
    strcpy(w.kw_text[0],"levi");prefix[0]=5;
    assert(bias_keywords(&w,prefix,1,actual,edge)==0);
    w.n_kw=ND_WH_MAX_KW;
    for(unsigned q=0;q<w.n_kw;q++) {
        w.kw_group[q]=q;strcpy(w.kw_text[q],"never said");
        w.kw_len[q]=1;w.kw[q][0]=q;
    }
    assert(bias_keywords(&w,NULL,0,actual,edge)==ND_WH_MAX_KW);
    for(unsigned q=0;q<w.n_kw;q++)actual[edge[q].token]=edge[q].original;
    assert(memcmp(actual,original,sizeof(actual))==0);
    for(unsigned g=0;g<8;g++)strcpy(w.kw_text[g],groups[g]);
    unsigned comparisons=0,duplicates=0,full=0;
    for(unsigned trial=0;trial<4096;trial++) {
        uint32_t ids[24],n=trial%25;
        for(unsigned i=0;i<n;i++)ids[i]=4+next_random()%8;
        w.n_kw=trial%(ND_WH_MAX_KW+1);
        if(trial%16==0)w.n_kw=ND_WH_MAX_KW;
        full+=w.n_kw==ND_WH_MAX_KW;
        const float bonus[]={0.0f,4.0f,-4.0f,0x1p-24f,1e10f};
        w.kw_bonus=bonus[trial%5];
        for(unsigned q=0;q<w.n_kw;q++) {
            w.kw_group[q]=q%8;w.kw_len[q]=1+next_random()%ND_WH_KW_TOKENS;
            for(unsigned k=0;k<w.kw_len[q];k++)w.kw[q][k]=4+next_random()%8;
            /* Different phrases may share a next token, including EOS. */
            if(trial%3==0)w.kw[q][0]=trial%VOCAB;
        }
        for(unsigned i=0;i<VOCAB;i++)original[i]=(int)(next_random()%129)-64;
        if(trial%7==0)for(unsigned i=0;i<VOCAB;i++)original[i]=0.0f;
        memcpy(expected,original,sizeof(original));memcpy(actual,original,sizeof(original));
        reference_bias_keywords(&w,ids,n,expected);
        keyword_undo undo[ND_WH_MAX_KW];
        unsigned touched=bias_keywords(&w,ids,n,actual,undo);
        assert(touched<=w.n_kw);
        assert(memcmp(expected,actual,sizeof(actual))==0);
        unsigned winner=0;
        for(unsigned i=1;i<VOCAB;i++)if(actual[i]>actual[winner])winner=i;
        for(unsigned i=0;i<touched;i++) {
            for(unsigned j=0;j<i;j++)assert(undo[i].token!=undo[j].token);
            actual[undo[i].token]=undo[i].original;
        }
        duplicates+=touched<w.n_kw;
        assert(memcmp(actual,original,sizeof(actual))==0);
        assert(memcmp(&actual[winner],&original[winner],sizeof(float))==0);
        comparisons++;
    }
    assert(duplicates && full);nd_tok_free(&w.tok);
    printf("PASS %u full-vocabulary comparisons: exact biased logits, winner, unbiased confidence input and restoration; %u full-capacity cases; undo=%zu bytes\n",comparisons,full,sizeof(keyword_undo)*ND_WH_MAX_KW);
    return 0;
}

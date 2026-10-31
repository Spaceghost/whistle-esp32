/* Research-only native admission, prefix and frontend probe. No device I/O. */
#include "nd_whistle.h"
#include "nd_quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void *read_file(const char *name,size_t *size) {
    FILE *f=fopen(name,"rb");void *data=NULL;
    if(!f)return NULL;
    if(fseek(f,0,SEEK_END) || ftell(f)<0)goto done;
    *size=(size_t)ftell(f);
    if(*size>64u*1024u*1024u || fseek(f,0,SEEK_SET))goto done;
    data=malloc(*size?*size:1);
    if(data && fread(data,1,*size,f)!=*size){free(data);data=NULL;}
done:fclose(f);return data;
}
static float row_logp(const float *row,unsigned n,unsigned best) {
    float maximum=row[0],sum=0.0f;
    for(unsigned k=1;k<n;k++)if(row[k]>maximum)maximum=row[k];
    for(unsigned k=0;k<n;k++)sum+=nd_expf(row[k]-maximum);
    return row[best]-(maximum+logf(sum));
}
int main(int argc,char **argv) {
    size_t bytes;void *blob;nd_whistle w={0};int rc=1;
    if(argc<3)return 2;
    blob=read_file(argv[1],&bytes);if(!blob)return 3;
    int opened=nd_whistle_open(&w,blob,bytes);
    if(opened){fprintf(stderr,"open %d\n",opened);nd_whistle_close(&w);free(blob);return 4;}
    if(!strcmp(argv[2],"inspect")) {
        printf("{\"vocabulary\":%u,\"model_state_bytes\":%zu}\n",w.vocab,sizeof(w));rc=0;
    }else if(!strcmp(argv[2],"cq-linear") && argc==6) {
        char *end=NULL;unsigned long index=strtoul(argv[3],&end,10);
        if(!*argv[3] || *end || index>=w.c.h4.num_tensors)goto done;
        const nd_tensor *tensor=&w.t[index];
        if(!nd_cq_q8_ok(tensor) || tensor->ndim!=2)goto done;
        size_t nbytes;float *input=read_file(argv[4],&nbytes);
        size_t row_bytes=(size_t)tensor->shape[1]*sizeof(float);
        if(!input || !row_bytes || !nbytes || nbytes%row_bytes || nbytes/row_bytes>32){free(input);goto done;}
        unsigned rows=(unsigned)(nbytes/row_bytes);
        size_t count=(size_t)rows*tensor->shape[0];
        if(count>4u*1024u*1024u){free(input);goto done;}
        float *output=malloc(count*sizeof(float));nd_q8_act activations={0};
        if(!output || nd_cq_q8_prepare(tensor,input,rows,&activations)){free(output);free(input);goto done;}
        const void *data=(const unsigned char*)blob+tensor->offset;
        nd_cq_q8_gemm(&w.c,tensor,data,&activations,0,tensor->shape[0],output);
        FILE *f=fopen(argv[5],"wbx");
        if(f){rc=fwrite(output,sizeof(float),count,f)==count?0:1;fclose(f);}
        nd_cq_q8_free(&activations);free(output);free(input);
    }else if(!strcmp(argv[2],"mel") && argc==5) {
        size_t nbytes;float *pcm=read_file(argv[3],&nbytes);
        if(!pcm || nbytes%sizeof(float) || nbytes>16000*30*sizeof(float)){free(pcm);goto done;}
        unsigned n=(unsigned)(nbytes/sizeof(float)),frames=n/w.hop;
        float *out=malloc((size_t)frames*w.mels*sizeof(float));if(!out){free(pcm);goto done;}
        nd_whistle_logmel(&w,pcm,n,out);
        FILE *f=fopen(argv[4],"wb");
        if(f){size_t want=(size_t)frames*w.mels;rc=fwrite(out,sizeof(float),want,f)==want?0:1;fclose(f);}
        free(out);free(pcm);
    }else if(!strcmp(argv[2],"decode-pcm") && argc==5) {
        size_t nbytes;float *pcm=read_file(argv[3],&nbytes);
        if(!pcm || nbytes%sizeof(float) || nbytes<160*sizeof(float) || nbytes>16000*30*sizeof(float)){free(pcm);goto done;}
        unsigned n=(unsigned)(nbytes/sizeof(float)),frames=nd_whistle_frames(&w,n),ids[320],prefix[322];
        float *enc=malloc((size_t)frames*w.d*sizeof(float));
        if(!enc){free(pcm);goto done;}
        if(nd_whistle_encode(&w,pcm,n,enc)){free(enc);free(pcm);goto done;}
        int count=nd_whistle_decode(&w,enc,frames,0,ids,320);
        if(count<0){free(enc);free(pcm);goto done;}
        float score=w.score;
        prefix[0]=2;
        const nd_tensor *manifest=&w.t[w.i_audio_cfg];
        prefix[1]=(unsigned)((const float*)((const unsigned char*)blob+manifest->offset))[2];
        memcpy(prefix+2,ids,(size_t)count*sizeof(unsigned));
        size_t logits_count=((size_t)count+2)*w.vocab;
        float *lg=malloc(logits_count*sizeof(float));
        if(!lg){free(enc);free(pcm);goto done;}
        if(nd_whistle_logits(&w,enc,frames,prefix,(unsigned)count+2,lg)){free(lg);free(enc);free(pcm);goto done;}
        char path[4096];
        int written=snprintf(path,sizeof(path),"%s.enc",argv[4]);
        FILE *f=written>0 && (size_t)written<sizeof(path)?fopen(path,"wbx"):NULL;
        rc=1;
        if(f){size_t size=(size_t)frames*w.d;if(fwrite(enc,sizeof(float),size,f)==size)rc=0;fclose(f);}
        written=snprintf(path,sizeof(path),"%s.logits",argv[4]);
        f=written>0 && (size_t)written<sizeof(path)?fopen(path,"wbx"):NULL;
        if(f){if(fwrite(lg,sizeof(float),logits_count,f)!=logits_count)rc=1;fclose(f);}else rc=1;
        printf("{\"vocabulary\":%u,\"prefix\":%u,\"score_hex\":\"%a\",\"tokens\":[",w.vocab,prefix[1],(double)score);
        for(int i=0;i<count;i++)printf("%s%u",i?",":"",ids[i]);
        puts("]}");free(lg);free(enc);free(pcm);
    }else if(!strcmp(argv[2],"prefix") && argc==4) {
        unsigned prefix=(unsigned)strtoul(argv[3],NULL,10),tokens[3]={2,prefix,0},ids[1];
        float *enc=calloc(w.d,sizeof(float)),*lg=malloc((size_t)3*w.vocab*sizeof(float));
        if(!enc || !lg){free(enc);free(lg);goto done;}
        if(nd_whistle_logits(&w,enc,1,tokens,2,lg)){free(enc);free(lg);goto done;}
        unsigned best=0;for(unsigned k=0;k<w.vocab;k++) {
            if(!isfinite(lg[w.vocab+k])){free(enc);free(lg);goto done;}
            if(lg[w.vocab+k]>lg[w.vocab+best])best=k;
        }
        float expected=row_logp(lg+w.vocab,w.vocab,best);
        if(best!=1) {
            tokens[2]=best;
            if(nd_whistle_logits(&w,enc,1,tokens,3,lg)){free(enc);free(lg);goto done;}
            unsigned next=0;
            for(unsigned k=1;k<w.vocab;k++)if(lg[2*w.vocab+k]>lg[2*w.vocab+next])next=k;
            expected=(expected+row_logp(lg+2*w.vocab,w.vocab,next))/2.0f;
        }
        int count=nd_whistle_decode(&w,enc,1,0,ids,1);
        int token_ok=(best==1 && count==0)||(best!=1 && count==1 && ids[0]==best);
        int score_ok=isfinite(w.score) && !memcmp(&expected,&w.score,sizeof(expected));
        rc=token_ok && score_ok?0:1;
        printf("{\"prefix\":%u,\"first_token_matches_teacher_forcing\":%s,\"confidence_matches_teacher_forcing\":%s,\"expected_score_hex\":\"%a\",\"actual_score_hex\":\"%a\"}\n",prefix,token_ok?"true":"false",score_ok?"true":"false",(double)expected,(double)w.score);
        free(enc);free(lg);
    }
done:nd_whistle_close(&w);free(blob);return rc;
}

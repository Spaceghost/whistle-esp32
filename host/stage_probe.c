/* Host-only access to static encoder stages for numerical diagnosis.
 * Including the implementation keeps production APIs/state unchanged. The
 * linker resolves its remaining dependencies from needle_engine, without
 * pulling a second copy of nd_whistle.c from that static archive. */
#include <stdio.h>
#include "../engine/src/nd_whistle.c"

static void *load(const char *path,size_t *size) {
    FILE *f=fopen(path,"rb");void *p=NULL;
    if(!f)return NULL;
    if(fseek(f,0,SEEK_END) || ftell(f)<0)goto done;
    *size=(size_t)ftell(f);
    if(*size>64u*1024u*1024u || fseek(f,0,SEEK_SET))goto done;
    p=malloc(*size?*size:1);
    if(p && fread(p,1,*size,f)!=*size){free(p);p=NULL;}
done:fclose(f);return p;
}
static int save(const char *root,const char *name,const float *data,size_t count) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s.f32",root,name);
    if(n<0 || (size_t)n>=sizeof(path))return 1;
    FILE *f=fopen(path,"wbx");if(!f)return 1;
    int rc=fwrite(data,sizeof(float),count,f)!=count;fclose(f);return rc;
}
int main(int argc,char **argv) {
    if(argc!=4 && argc!=5)return 2;
    if(argc==5){if(strcmp(argv[4],"--float"))return 2;nd_cq_int8=0;}
    size_t bytes,nbytes;void *blob=load(argv[1],&bytes);
    float *pcm=load(argv[2],&nbytes),*mel=NULL,*s=NULL,*lanes=NULL,*out=NULL;
    nd_whistle w={0};int rc=1;
    if(!blob || !pcm || !nbytes || nbytes%sizeof(float) || nbytes>30u*16000u*sizeof(float))goto done;
    if(nd_whistle_open(&w,blob,bytes))goto done;
    unsigned n=(unsigned)(nbytes/sizeof(float)),M=n/w.hop,T=nd_whistle_frames(&w,n);
    if(!M || !T)goto done;
    mel=malloc((size_t)M*w.mels*sizeof(float));s=malloc((size_t)T*w.d*sizeof(float));
    lanes=malloc((size_t)T*w.lanes*w.d*sizeof(float));out=malloc((size_t)T*w.d*sizeof(float));
    if(!mel || !s || !lanes || !out)goto done;
    nd_whistle_logmel(&w,pcm,n,mel);
    if(save(argv[3],"features",mel,(size_t)M*w.mels))goto done;
    stem(&w,mel,M,s);
    if(save(argv[3],"stem",s,(size_t)T*w.d))goto done;
    lanes_init(s,T,w.lanes,w.d,lanes);
    for(unsigned i=0;i<w.enc_layers;i++) {
        char name[64];
        mhc_step(&w,w.i_enc_mhc,i,i,lanes,T,enc_block,NULL);
        snprintf(name,sizeof(name),"lanes-%u",i);
        if(save(argv[3],name,lanes,(size_t)T*w.lanes*w.d))goto done;
    }
    lanes_out(&w,lanes,T,w.i_enc_final,out);
    if(save(argv[3],"encoder",out,(size_t)T*w.d))goto done;
    rc=0;
done:nd_whistle_close(&w);free(blob);free(pcm);free(mel);free(s);free(lanes);free(out);return rc;
}

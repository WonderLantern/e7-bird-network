/* Host-only finite64 scalar reference for the decoded 744 consumer.
 * No network or target binary execution. Explicit float32/RNE candidate.
 * The linked archived qsort retains its original BSD notice in its own file.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <float.h>

typedef struct { float x,y,w,h; uint16_t score,cls; } Box;
typedef struct {
    uint32_t visited,accepted,outputs,error_ordinal,error_field;
    uint16_t initial[24];
    uint32_t ordinals[32];
    Box candidates[32];
    uint16_t result[16][6];
} Report;
_Static_assert(sizeof(float)==4 && FLT_MANT_DIG==24,"binary32 required");
_Static_assert(sizeof(Box)==20 && offsetof(Box,score)==16,"native box layout");
void archived_qsort(void*,size_t,size_t,int(*)(const void*,const void*));

static uint16_t u16(const uint8_t *p) { return p[0]|((uint16_t)p[1]<<8); }
static uint32_t u32(const uint8_t *p) {
    return p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static float as_float(uint32_t bits) { float f; memcpy(&f,&bits,4); return f; }
static uint32_t as_bits(float f) { uint32_t u; memcpy(&u,&f,4); return u; }
/* Volatile assignment enforces a float rounding point for each primitive. */
static float add(float a,float b) { volatile float r=a+b; return r; }
static float mul(float a,float b) { volatile float r=a*b; return r; }
static float divide(float a,float b) { volatile float r=a/b; return r; }
static float half(uint16_t u) {
    int e=(u>>10)&31, m=u&1023;
    float f=ldexpf((float)(e?m+1024:m),e?e-25:-24);
    return u&32768?-f:f;
}
static float polynomial(float x) {
    int32_t n=(int32_t)mul(x,as_float(0x3fb8aa3b));
    float r=add(x,-mul((float)n,as_float(0x3f317218)));
    float r2=mul(r,r), r4=mul(r2,r2);
    float high=add(mul(r,as_float(0x394d4a8b)),as_float(0x3ab91bcb));
    float middle=add(mul(r,as_float(0x3c0897a6)),as_float(0x3d2aa390));
    float low=add(mul(r,as_float(0x3e2aaa6a)),as_float(0x3f00000a));
    float linear=add(mul(r,as_float(0x3f800001)),1.0f);
    float p=add(mul(add(mul(high,r2),middle),r4),add(mul(low,r2),linear));
    return as_float(as_bits(p)+((uint32_t)n<<23));
}
static float sigmoid(float x) { return divide(1.0f,add(1.0f,polynomial(-x))); }
__declspec(dllexport) int reference_mode(void) { return fegetround()==FE_TONEAREST; }
__declspec(dllexport) int reference_numeric(uint16_t h,uint32_t *expbits,uint32_t *sigbits) {
    if ((h&32767)>0x5400 || !reference_mode()) return -1;
    float v=half(h); *expbits=as_bits(polynomial(v)); *sigbits=as_bits(sigmoid(v)); return 0;
}
static int rank(const uint8_t *curve,float v,int low) {
    int high=999;
    if(low>high)return 1000;
    while(low<=high) {
        int m=(low+high)/2;
        float item=as_float(u32(curve+4*m));
        if(v>item) low=m+1;
        else if(item>v) high=m-1;
        else return m;
    }
    return high+1>1000?1000:high+1;
}
static int compare(const void *a,const void *b) {
    return (int)((const Box*)b)->score-(int)((const Box*)a)->score;
}
static uint16_t read_head(const uint8_t *p,int stage,int anchor,int field,int row,int col) {
    int stride=stage?32:48, rows=stage?8:16;
    int offset=(stage?0x4200:0)+(anchor*11+field)*stride*rows+row*stride+2*col;
    return u16(p+offset);
}
static int consumed(const uint8_t *p,int st,int an,int field,int row,int col,float *v) {
    uint16_t h=read_head(p,st,an,field,row,col);
    if((h&32767)>0x5400)return -1;
    *v=half(h); return 0;
}
/* Wide finite-logit profile: explicit saturation and modulo32 geometry.
 * No undefined signed overflow or out-of-range float-to-int casts. */
static int32_t signed32(uint32_t v) { int32_t r;memcpy(&r,&v,4);return r; }
static int32_t plus32(int32_t a,int32_t b) { return signed32((uint32_t)a+(uint32_t)b); }
static int32_t minus32(int32_t a,int32_t b) { return signed32((uint32_t)a-(uint32_t)b); }
static int32_t coordinate(float v,int scale) {
    float scaled=mul(v,(float)scale);
    float biased=add(scaled,scaled>=0?0.5f:-0.5f);
    if(biased>=2147483648.0f)return INT32_MAX;
    if(biased<=-2147483648.0f)return INT32_MIN;
    return (int32_t)biased; /* finite in this profile; range checked above */
}
static int clip(Box b,uint16_t *o) {
    int32_t x=coordinate(b.x,384),y=coordinate(b.y,256);
    int32_t w=coordinate(b.w,384),h=coordinate(b.h,256);
    int32_t l=minus32(x,w/2),t=minus32(y,h/2),r=plus32(l,w),bot=plus32(t,h);
    if(r>383) {w=minus32(w,minus32(r,383));x=plus32(minus32(383,w),w/2);}
    if(l<0) {w=plus32(w,l);x=w/2;}
    if(bot>255) {h=minus32(h,minus32(bot,255));y=plus32(minus32(255,h),h/2);}
    if(t<0) {h=plus32(h,t);y=h/2;}
    if((uint32_t)x>383||((uint32_t)w-1)>383||(uint32_t)y>255||
       ((uint32_t)h-1)>255||b.score>1000) return 0;
    o[0]=x;o[1]=y;o[2]=w;o[3]=h;o[4]=b.score;o[5]=b.cls;return 1;
}
__declspec(dllexport) int reference_consume(const uint8_t *heads,size_t length,
                        const uint8_t *ipu,size_t ipulen,Report *out) {
    if(length!=0x6300||ipulen!=0x80000||!reference_mode())return -1;
    memset(out,0,sizeof(*out));
    float objgate=as_float(u32(ipu+0x66578)), scoregate=as_float(u32(ipu+0x6657c));
    for(int k=0;k<24;k++) {
        int cls=(u16(ipu+0x4ee48+2*k)>>8)&15;if(cls>5)cls=5;
        out->initial[k]=rank(ipu+0x4ee78+4000*k,scoregate,u16(ipu+0x66584+2*cls)-1);
    }
    unsigned ordinal=0;
    for(int st=0;st<2;st++) for(int an=0;an<(st?3:2);an++)
    for(int row=0;row<(st?8:16);row++) for(int col=0;col<(st?12:24);col++,ordinal++) {
        float v;
        out->visited++;out->error_ordinal=ordinal;out->error_field=4;
        if(consumed(heads,st,an,4,row,col,&v))return -2;
        float obj=sigmoid(v);
        if(objgate>obj)continue;
        int token=(st<<4)+(st?2:0)+an, best=0,cls=0;
        for(int c=0;c<6;c++) {
            int index=(c<<8)+token,k;
            for(k=0;k<24;k++)if(u16(ipu+0x4ee48+2*k)==index)break;
            if(k==24)continue;
            out->error_field=5+c;
            if(consumed(heads,st,an,5+c,row,col,&v))return -2;
            float combined=mul(sigmoid(v),obj);
            if(scoregate>combined)continue;
            int r=rank(ipu+0x4ee78+4000*k,combined,out->initial[k]-1);
            if(r>best){best=r;cls=c;}
        }
        if(!best)continue;
        float g[4];
        for(int c=0;c<4;c++) {
            out->error_field=c;
            if(consumed(heads,st,an,c,row,col,g+c))return -2;
        }
        int a=(st?2:0)+an;
        Box b;
        b.x=divide(add(sigmoid(g[0]),(float)col),st?12.0f:24.0f);
        b.y=divide(add(sigmoid(g[1]),(float)row),st?8.0f:16.0f);
        b.w=divide(mul((float)u16(ipu+0x665a2+4*a),polynomial(g[2])),384.0f);
        b.h=mul(mul((float)u16(ipu+0x665a4+4*a),polynomial(g[3])),as_float(0x3b800000));
        b.score=best;b.cls=cls;
        out->ordinals[out->accepted]=ordinal;
        out->candidates[out->accepted++]=b;
        if(out->accepted==32)goto finalize;
    }
finalize:;
    Box sorted[32];memcpy(sorted,out->candidates,out->accepted*sizeof(Box));
    float factor=as_float(u32(ipu+0x66580));
    for(unsigned i=0;i<out->accepted;i++) {
        Box *a=sorted+i;if(!a->score)continue;
        for(unsigned j=i+1;j<out->accepted;j++) {
            Box *b=sorted+j;if(!b->score)continue;
            if(a->cls!=b->cls&&!((a->cls==0&&b->cls==4)||(a->cls==4&&b->cls==0)))continue;
            float dx=fabsf(add(a->x,-b->x)),dy=fabsf(add(a->y,-b->y));
            float tx=mul(add(mul(factor,a->w),mul(factor,b->w)),0.5f);
            float ty=mul(add(mul(factor,a->h),mul(factor,b->h)),0.5f);
            if(!(tx>dx&&ty>dy))continue;
            if((a->cls==0&&b->cls==4)||(a->cls==b->cls&&b->score>a->score)) {
                a->score=0;break;
            }
            b->score=0;
        }
    }
    unsigned active=0;for(unsigned i=0;i<out->accepted;i++)active+=sorted[i].score!=0;
    archived_qsort(sorted,out->accepted,sizeof(Box),compare);
    for(unsigned i=0;i<active&&out->outputs<16;i++)
        if(clip(sorted[i],out->result[out->outputs]))out->outputs++;
    out->error_ordinal=out->error_field=0;return 0;
}

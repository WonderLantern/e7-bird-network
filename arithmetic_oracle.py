"""Independent exact integer arithmetic oracle retained from R13."""
def quantize(n,k,sign):
    # Independent exact integer n*2**k -> half RTZ, flush subnormal output.
    n=abs(n)
    if n==0:return sign
    exponent=n.bit_length()-1+k
    if exponent < -14:return sign
    if exponent > 15:return sign|0x7bff
    shift=n.bit_length()-11
    mantissa=(n>>shift) if shift>=0 else (n<<-shift)
    return sign|((exponent+15)<<10)|(mantissa&1023)

def parts(bits):
    e=(bits>>10)&31
    return max(e,1),(bits&1023)|(1024 if e else 0),(-1 if bits&32768 else 1)

def oracle_add(a,b):
    ea,ma,sa=parts(a);eb,mb,sb=parts(b);e=max(ea,eb)
    n=sa*((ma*16)>>(e-ea))+sb*((mb*16)>>(e-eb))
    larger=a if (a&32767)>(b&32767) else b
    sign=(32768 if n<0 else 0) if n else (larger&32768)
    return quantize(n,e-29,sign)

def oracle_mul(a,b):
    ea,ma,sa=parts(a);eb,mb,sb=parts(b)
    return quantize(ma*mb,ea+eb-50,(a^b)&32768)

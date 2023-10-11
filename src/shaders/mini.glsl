#version 430
layout(binding = 0) uniform sampler2D backBuffer0;
layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
vec2 resolution = vec2(SCREEN_XRESO, SCREEN_YRESO);
#define NUM_SAMPLES_PER_SEC 48000.
float time = waveOutPosition / NUM_SAMPLES_PER_SEC;
#else
layout(location = 2) uniform float time;
layout(location = 3) uniform vec2 resolution;
#endif
out vec4 outColor0;
#define LoopMax 256
#define LenMax 100.0
#define NormalEPS 0.001
#define DistMin 0.001
float Time;
int MatID;
const float PI = acos(-1.), TAU = PI * 2., GOLD = PI * (3. - sqrt(5.));
const vec3 k2000 = vec3(255, 137, 14) / 255., k12000 = vec3(191, 211, 255) / 255.;
#define remap(x,a,b,c,d)((((x)-(a))/((b)-(a)))*((d)-(c))+(c))
#define remapc(x,a,b,c,d)clamp(remap(x,a,b,c,d),min(c,d),max(c,d))
#define saturate(x)clamp(x,0.0,1.0)
#define linearstep(a,b,x)min(max(((x)-(a))/((b)-(a)),0.0),1.0)
#define opRepLim(p,c,l)((p)-(c)*clamp(round((p)/(c)),-(l),(l)))
#define opRepLimID(p,c,l)(clamp(round((p)/(c)),-(l),(l))+(l))
vec2 orbit(float t)
{
    return vec2(cos(t), sin(t));
}
mat2 rot(float x)
{
    vec2 v = orbit(x);
    return mat2(v.x, v.y, -v.y, v.x);
}
bool tl(float intime, float outtime, out float lt)
{
    lt = (Time - intime) / (outtime - intime);
    return 0. <= lt && lt < 1.;
}
vec3 pcg33(vec3 v)
{
    uvec3 x = floatBitsToUint(v);
    x = (x >> 8U ^ x.yzx) * 1103515245u;
    x = (x >> 8U ^ x.yzx) * 1103515245u;
    x = (x >> 8U ^ x.yzx) * 1103515245u;
    return vec3(x) / float(-1u);
}
vec3 perlin32(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3. - 2. * f);
    vec3 vx0 = mix(pcg33(vec3(i, .12)), pcg33(vec3(i + vec2(1, 0), .12)), f.x), vx1 = mix(pcg33(vec3(i + vec2(0, 1), .12)), pcg33(vec3(i + vec2(1), .12)), f.x);
    return mix(vx0, vx1, f.y);
}
vec3 fbm32(vec2 p)
{
    const mat2 R = rot(GOLD) * 2.;
    float a = 1.;
    vec4 v = vec4(0);
    for(int i = 0; i < 6; i++) v += a * vec4(perlin32(p), 1), a *= .5, p *= R;
    return v.xyz / v.w;
}
float pow5(float x)
{
    return x * x * (x * x) * x;
}
vec3 Microfacet_BRDF(vec3 albedo, float metallic, float paramRoughness, vec3 L, vec3 V, vec3 N)
{
    float roughness = paramRoughness * paramRoughness;
    roughness = max(roughness, .001);
    vec3 f0 = .04 * (1. - metallic) + albedo * metallic, H = normalize(L + V);
    float NoV = saturate(dot(N, V)) + 1e-5, NoL = saturate(dot(N, L)), NoH = saturate(dot(N, H)), LoH = saturate(dot(L, H)), da = NoH * roughness, dk = roughness / (1. - NoH * NoH + da * da), va = roughness * roughness, f90 = .5 + 2. * roughness * LoH * LoH;
    return (dk * dk / PI * (.5 / (NoL * sqrt(max(0., NoV * NoV * (1. - va) + va)) + NoV * sqrt(max(0., NoL * NoL * (1. - va) + va)))) * (f0 + (1. - f0) * pow5(1. - LoH)) + albedo * (1. - metallic) * ((1. + (f90 - 1.) * pow5(1. - NoL)) * (1. + (f90 - 1.) * pow5(1. - NoV)) / PI)) * NoL;
}
float fOpUnionRound(float a, float b, float r)
{
    return min(a, b);
}
float fOpDifferenceRound(float a, float b, float r)
{
    return max(a, -b);
}
float sdBox(vec3 p, vec3 b)
{
    vec3 d = abs(p) - b * .5;
    return min(max(d.x, max(d.y, d.z)), 0.) + length(max(d, 0.));
}
const vec2 RoomSize = vec2(6, 4) * .5;
vec2 opRoomRep(inout vec3 p)
{
    vec2 ip = floor(p.xy / RoomSize) * RoomSize + RoomSize * .5;
    p.xy = mod(p.xy, RoomSize) - .5 * RoomSize;
    return ip;
}
float sdf(vec3 p)
{
    vec3 op = p;
    vec2 ip = opRoomRep(p);
    vec3 h3 = pcg33(vec3(ip, 0));
    float td, td2;
    vec3 tp;
    bool isfloorB = fract(.5 * ip.x / RoomSize.x) < .5;
    float isfloor = float(!isfloorB) * 1e9, isroom = float(isfloorB) * 1e9;
    tp = abs(fract(op * 10.) - .5);
    float mizo = dot(vec3(1), smoothstep(.04, .02, tp)) / 3. * .001;
    const float slope = RoomSize.x * .2;
    float cy = linearstep(-RoomSize.x * .5 + slope, RoomSize.x * .5 - slope, p.x) * RoomSize.y + RoomSize.y * .25, fy = p.y - cy, fd = min(min(max((min(abs(fy), abs(fy + RoomSize.y)) - RoomSize.y * .25) * (slope * 2. / sqrt(RoomSize.y * RoomSize.y + slope * slope * 4.)), -p.z), -p.z + 3.), max(abs(p.x) - RoomSize.x * .5 + slope, -p.z + .5));
    fd += mizo;
    tp = p - vec3(-RoomSize.x * .45, 0, -.07);
    tp.x = abs(tp.x - .1) - .1;
    fd = min(fd, length(tp.xz) - .07);
    tp.y = abs(fract(tp.y) - .5) - .06;
    fd = min(fd, sdBox(tp + vec3(.1, 0, 0), vec3(.05, 0, 0)) - .01);
    vec2 dd = abs(vec2(length(tp.xz), tp.y)) - vec2(.08, .02);
    fd = min(fd, min(max(dd.x, dd.y), 0.) + length(max(dd, 0.)));
    fd += isfloor;
    float hd = -p.z;
    hd = fOpDifferenceRound(hd, sdBox(p, vec3(RoomSize - .2, 4)), .01);
    tp = p - vec3(0, 0, .75);
    tp.x = abs(tp.x);
    const vec2 rsize = RoomSize - .2;
    hd = fOpUnionRound(hd, sdBox(tp, vec3(rsize, .05)), .01);
    const vec3 msize = vec3(rsize.x * .25 - .125, rsize.y * .75, .1);
    hd = fOpDifferenceRound(hd, sdBox(tp - vec3(msize.x * .5 + .05, -msize.y * .1, 0), msize), .01);
    const vec3 msize2 = vec3(msize.x, msize.y * .5, msize.z);
    hd = fOpDifferenceRound(hd, sdBox(tp - vec3(msize2.x * 1.5 + .15, msize2.y * .3, 0), msize2), .01);
    hd = min(hd, -.01 + sdBox(p - vec3(0, -RoomSize.y * .25, .1), vec3(RoomSize.x - .2, RoomSize.y * .3, .025)));
    tp = p - vec3(0, -RoomSize.y * .25, .1);
    tp.x = opRepLim(tp.x, .2, 6);
    hd = fOpDifferenceRound(hd, sdBox(tp, vec3(.06, RoomSize.y * .25, .05)), .01);
    hd = min(hd, -.01 + sdBox(p - vec3(0, -RoomSize.y * .1 - .025, 0), vec3(RoomSize.x - .2, .05, .2)));
    td = -.01 + sdBox(p - vec3(1.1, -.77, -.02), vec3(.4, .28, .1));
    tp = p - vec3(1.1, -.77, -.02) - vec3(-.06, 0, -.1);
    td = fOpDifferenceRound(td, sdBox(tp, vec3(0, 0, .1)) - .1, .005);
    tp.z -= .05;
    td = min(td, length(tp) - .015);
    const float div = PI / 32.;
    float a = atan(tp.y, tp.x);
    a = mod(a, div * 2.) - div;
    tp.xy = orbit(a) * length(tp.xy);
    td = fOpUnionRound(td, sdBox(tp, vec3(.2, 0, 0)) - .002, .005);
    hd = min(hd, td);
    hd += isroom;
    float d = min(hd, fd), shimaru = smoothstep(0., 1.5, .9 - p.y);
    vec3 kp = p;
    kp.x *= mix(1.1, 1., shimaru);
    float shf = h3.x * TAU, lt = Time * .5 + shf, kz = cos(shf + kp.x * PI * 7. + .5 * PI * cos(kp.x * 7.)) * .05 + cos(kp.x * 2. + lt * 2. + .5 * PI * cos(lt)) * .05;
    kp -= vec3(0, 0, 1. + kz * shimaru);
    td = sdBox(kp, vec3(RoomSize - .4, .01)) * .5 - .01;
    td += isroom;
    MatID = td < d ? 1 : 0;
    d = min(d, td);
    td2 = length(p - vec3(sign(h3.x - .5), -.09, 0)) - .1;
    td2 += isroom;
    td = length(p - vec3(-RoomSize * .5 + .1, 2.9)) - .1;
    td += isfloor;
    td = min(td, td2);
    MatID = td < d ? 2 : MatID;
    return min(d, td);
}
vec2 march(vec3 rd, vec3 ro, out vec3 rp)
{
    float v = 1., ph = LenMax, dist, len = 0.;
    for(int i = 0; i < LoopMax; i++)
    {
        rp = ro + rd * len;
        float _lt;
        if(tl(90., 105., _lt))
            rp.z = abs(rp.z) - 2.;
        if(tl(105., 120., _lt))
            rp.xz = vec2((atan(rp.z, rp.x) + PI) / TAU * RoomSize.x * 8., length(rp.xz)), rp.z -= 4.;
        dist = sdf(rp);
        float y = dist * dist / (2. * ph), d = sqrt(dist * dist - y * y);
        v = min(v, d / (.02 * max(0., len - y)));
        ph = dist;
        vec2 irp = (floor((rp.xy + sign(rd.xy) * RoomSize) / RoomSize) + .5) * RoomSize, bd = abs(irp - rp.xy) - .5 * RoomSize;
        bd = max(bd, 0.) / abs(rd.xy) + DistMin + float(rp.z < -2.) * LenMax;
        dist = min(dist, min(bd.x, bd.y));
        len += dist;
        if(dist < DistMin)
            return vec2(1, .1);
        if(len > LenMax)
            return vec2(0, max(v, .1));
    }
    return vec2(0, max(v, .1));
}
const vec3 DirectionalLight = normalize(vec3(1, 1, -1));
vec3 sky(vec3 rd)
{
    vec2 th = vec2(atan(rd.x, -rd.z), acos(rd.y) * 2.) / TAU * 20. + vec2(.05, .01) * Time;
    vec3 scol = vec3(mix(.1, .5, fbm32(th).x));
    float up = saturate((rd.y + .5) * .5);
    vec3 col = mix(vec3(.01), scol, up);
    float sun = saturate(dot(rd, DirectionalLight));
    sun = smoothstep(.8, 1., sun);
    col *= mix(1., 3., sun);
    col = mix(col, vec3(1), pow(linearstep(.995, 1., sun * sun), 5.));
    col *= linearstep(1., .9, rd.y);
    return col;
}
vec3 shading(inout vec3 P, vec3 V, vec3 N)
{
    vec3 tN = sign(N) * abs(N);
    tN /= dot(vec3(1), tN);
    vec2 uv = tN.x * P.zy + tN.y * P.xz + tN.z * P.xy;
    vec3 RP = P;
    vec2 ip = opRoomRep(RP);
    vec3 h3 = pcg33(vec3(.1, ip));
    float type = 0.;
    vec3 albedo = vec3(1);
    float roughness = .5, metallic = .5;
    vec3 plcol = (step(h3.y, .9) * .8 + .2) * mix(k2000, k12000, h3.x);
    if(MatID == 0)
    {
        vec3 fbm = fbm32(uv * 3. * vec2(3, 1)), fbm2 = fbm32(uv * 96. * vec2(2, 1));
        albedo = vec3(saturate(1.3 * mix(.6, 1., fbm.y) * mix(.8, 1., pow(fbm2.x, 3.))));
        roughness = mix(.5, 1., pow(fbm.y, 3.));
        metallic = .01;
        N = normalize(N + (fbm * 2. - 1.) * .05);
    }
    else if(MatID == 1)
    {
        float h = h3.x;
        albedo = h < .7 ? vec3(.8, .7, .6) : h < .8 ? vec3(.8, .2, .2) : h < .9 ? vec3(.8, .6, .3) : vec3(.5, .7, .8);
        roughness = .99;
        metallic = .01;
    }
    else
        type = 1., albedo = plcol;
    P += N * DistMin * 2.;
    vec3 _rp;
    vec2 sh = march(DirectionalLight, P, _rp);
    vec3 col = vec3(0);
    col += sh.y * Microfacet_BRDF(albedo, metallic, roughness, DirectionalLight, V, N) * k12000;
    vec3 L = (fract(.5 * ip.x / RoomSize.x) < .5 ? vec3(-RoomSize * .5 + .1, 2.9) : vec3(sign(h3.x - .5), -.06, 0)) - RP;
    float l = length(L);
    L /= l;
    col += plcol * pow(1. / (1. + max(0., l)), 2.) * Microfacet_BRDF(albedo, metallic, roughness, L, V, N);
    col *= sqrt(saturate(sdf(P + N * .05) / .05));
    return mix(col, albedo, type);
}
const vec2 FC = gl_FragCoord.xy, RES = resolution.xy, ASP = RES / min(RES.x, RES.y);
vec2 FocalUV = vec2(.5);
void getRORD(out vec3 ro, out vec3 rd, out vec3 dir, vec2 suv)
{
    float fov = 60., fisheye = 0.;
    ro = vec3(1);
    dir = vec3(0, 0, 1);
    float lt = 0.;
    if(tl(0., 15., lt))
        ro = vec3(.2, .2, mix(2.2, 1.5, lt)), dir = normalize(mix(vec3(.5, 1.5, 1), vec3(.1, .1, 1), lt));
    else if(tl(15., 30., lt))
        fov = mix(60., 90., lt), fisheye = mix(0., .4, lt), ro = mix(vec3(.4, .2, 2), vec3(.35, .6, .4), lt), dir = normalize(mix(vec3(-.5, -1, -1), vec3(0, 0, -1), lt));
    else if(tl(30., 45., lt))
        ro = vec3(mix(.8 - RoomSize.x, -.1, lt), -1, 0), dir = normalize(vec3(-1, -.1, .3));
    else if(tl(45., 60., lt))
        ro = vec3(mix(-.5, .2 - RoomSize.x, lt), -1.8, -.3), dir = normalize(vec3(1, .2, .8));
    else if(tl(60., 75., lt))
    {
        float z = floor(lt * 3.) * 8. + fract(lt * 3.);
        ro = vec3(0, 0, -1. - z);
        dir = vec3(0, 0, 1);
    }
    else if(tl(77., 90., lt))
    {
        float t = 20. * lt;
        vec3 ta = vec3(-RoomSize.x * .5 + .3, t * RoomSize.y, 0);
        ro = ta + vec3(mix(3, -3, lt), -5, -3);
        dir = normalize(ta - ro);
    }
    else if(tl(90., 105., lt))
        ro = vec3(-Time * 4., 0, 0), dir = normalize(vec3(-1, mix(.2, -.2, lt), 0));
    else if(tl(105., 120., lt))
        ro = vec3(0, Time * 2., 0), dir = normalize(vec3(0, 1, mix(0., 1., lt))), dir.xz *= rot(PI * lt);
    vec3 tebure = (fbm32(vec2(-4.2, Time * .1)) * 2. - 1.) * .05;
    ro += tebure;
    vec3 N = vec3(0, 1, 0), B = normalize(cross(N, dir));
    N = normalize(cross(dir, B));
    mat3 bnt = mat3(B, N, dir);
    float zf = 1. / tan(fov * PI / 360.);
    zf -= zf * length(suv) * fisheye;
    rd = normalize(bnt * vec3(suv, zf));
}
vec4 tracer(vec3 rd, vec3 ro)
{
    vec3 rp;
    if(march(rd, ro, rp).x < .5)
        return vec4(sky(rd), LenMax);
    float depth = length(rp - ro);
    const float h = NormalEPS;
    const vec2 k = vec2(1, -1);
    int mid = MatID;
    vec3 N = normalize(k.xyy * sdf(rp + k.xyy * h) + k.yyx * sdf(rp + k.yyx * h) + k.yxy * sdf(rp + k.yxy * h) + k.xxx * sdf(rp + k.xxx * h));
    MatID = mid;
    vec3 col = shading(rp, -rd, N);
    return vec4(col, depth);
}
vec3 acesFilm(vec3 x)
{
    return x * (2.51 * x + .03) / (x * (2.43 * x + .59) + .14);
}
vec3 postprocess(vec3 col, vec3 seed)
{
    col += pcg33(seed) * .05;
    col = pow(col, vec3(.8));
    col = acesFilm(col);
#define COG(_s,_b)col._s=smoothstep(0.0-(_b)*0.5,1.0+(_b)*0.5,col._s)
    COG(r, .05);
    float lt = 0.;
    if(tl(0., 10., lt))
        col *= lt * lt;
    else if(tl(14., 16., lt))
        col *= 2. * abs(lt - .5);
    else if(tl(29., 31., lt))
        col *= 2. * abs(lt - .5);
    else if(tl(44., 46., lt))
        col *= 2. * abs(lt - .5);
    else if(tl(59., 61., lt))
        col *= 2. * abs(lt - .5);
    else if(tl(74., 77., lt))
        col *= saturate(75. - Time);
    else if(tl(119., 120., lt))
        col *= 1. - lt;
    else if(Time >= 120.)
        col *= 0.;
    return col;
}
void main()
{
    Time = time;
    vec2 uv = FC / RES;
    uv += (pcg33(vec3(FC, Time)).xy - .5) * .25 / RES;
    vec2 suv = (uv - .5) * 2. * ASP;
    vec3 ro, rd, dir;
    getRORD(ro, rd, dir, suv);
    vec4 traced = tracer(rd, ro);
    traced.xyz = postprocess(traced.xyz, vec3(-Time, 4.2 * uv));
    vec4 back = texture(backBuffer0, uv);
    outColor0 = mix(traced, back, .5);
}
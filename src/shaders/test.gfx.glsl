#version 430

// .%%..%%..%%..%%..%%%%%%..%%%%%%...%%%%...%%%%%...%%...%%.
// .%%..%%..%%%.%%....%%....%%......%%..%%..%%..%%..%%%.%%%.
// .%%..%%..%%.%%%....%%....%%%%....%%..%%..%%%%%...%%.%.%%.
// .%%..%%..%%..%%....%%....%%......%%..%%..%%..%%..%%...%%.
// ..%%%%...%%..%%..%%%%%%..%%.......%%%%...%%..%%..%%...%%.
// .........................................................
// 

layout(binding = 0) uniform sampler2D backBuffer0;
// layout(binding = 1) uniform sampler2D backBuffer1;
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
// layout(location = 0) out vec4 outColor0;
// layout(location = 1) out vec4 outColor1;

// 
// ..%%%%...%%.......%%%%...%%%%%....%%%%...%%.....
// .%%......%%......%%..%%..%%..%%..%%..%%..%%.....
// .%%.%%%..%%......%%..%%..%%%%%...%%%%%%..%%.....
// .%%..%%..%%......%%..%%..%%..%%..%%..%%..%%.....
// ..%%%%...%%%%%%...%%%%...%%%%%...%%..%%..%%%%%%.
// ................................................
// 
#define LoopMax 256
#define LenMax 100.0
#define NormalEPS 0.001
#define DistMin 0.001
// #define DistMin 0.005

float Time;
int MatID;

// 
// ..%%%%...%%%%%%..%%..%%..%%%%%%..%%%%%....%%%%...%%.....
// .%%......%%......%%%.%%..%%......%%..%%..%%..%%..%%.....
// .%%.%%%..%%%%....%%.%%%..%%%%....%%%%%...%%%%%%..%%.....
// .%%..%%..%%......%%..%%..%%......%%..%%..%%..%%..%%.....
// ..%%%%...%%%%%%..%%..%%..%%%%%%..%%..%%..%%..%%..%%%%%%.
// ........................................................
// 
const float PI = acos(-1.0);
const float TAU = 2.0 * PI;
const float GOLD = PI * (3.0 - sqrt(5.0));// 2.39996...
const vec3 k2000 = vec3(255, 137, 14) / 255.0;
const vec3 k12000 = vec3(191, 211, 255) / 255.0;

#define remap(x,a,b,c,d) ((((x)-(a))/((b)-(a)))*((d)-(c))+(c))
#define remapc(x,a,b,c,d) clamp(remap(x,a,b,c,d),min(c,d),max(c,d))
#define saturate(x) clamp(x,0.0,1.0)
#define linearstep(a, b, x) min(max(((x) - (a)) / ((b) - (a)), 0.0), 1.0)
#define opRepLim(p,c,l) ((p)-(c)*clamp(round((p)/(c)),-(l),(l)))
#define opRepLimID(p,c,l) (clamp(round((p)/(c)),-(l),(l))+(l))

vec2 orbit(float a)
{
    return vec2(cos(a), sin(a));
}

mat2 rot(float x)
{
    vec2 v = orbit(x);
    return mat2(v.x, v.y, -v.y, v.x);
}

bool tl(float intime, float outtime, out float lt)
{
    lt = (Time - intime) / (outtime - intime);
    return (0.0 <= lt && lt < 1.0);
}

// 
// .%%..%%...%%%%...%%%%%%...%%%%...%%%%%%.
// .%%%.%%..%%..%%....%%....%%......%%.....
// .%%.%%%..%%..%%....%%.....%%%%...%%%%...
// .%%..%%..%%..%%....%%........%%..%%.....
// .%%..%%...%%%%...%%%%%%...%%%%...%%%%%%.
// ........................................
// 

vec3 pcg33(vec3 v)
{
    uvec3 x = floatBitsToUint(v);
    const uint k = 1103515245u;
    x = ((x >> 8U) ^ x.yzx) * k;
    x = ((x >> 8U) ^ x.yzx) * k;
    x = ((x >> 8U) ^ x.yzx) * k;
    return vec3(x) / float(-1u);
}
vec3 perlin32(vec2 p)
{
    const float magic = 0.12;
    vec2 i = floor(p);
    vec2 f = fract(p);
    // smoothstep
    f = f * f * (3.0 - 2.0 * f);

    vec3 vx0 = mix(pcg33(vec3(i, magic)), pcg33(vec3(i + vec2(1, 0), magic)), f.x);
    vec3 vx1 = mix(pcg33(vec3(i + vec2(0, 1), magic)), pcg33(vec3(i + vec2(1, 1), magic)), f.x);
    return mix(vx0, vx1, f.y);
}
vec3 fbm32(vec2 p)
{
    const int N = 6;
    const mat2 R = rot(GOLD) * 2.0;

    float a = 1.0;
    vec4 v = vec4(0);
    for(int i = 0; i < N; i++)
    {
        v += a * vec4(perlin32(p), 1);
        a *= 0.5;
        p *= R;
    }
    return v.xyz / v.w;
}

// 
// .%%%%%...%%%%%...%%%%%...%%%%%%.
// .%%..%%..%%..%%..%%..%%..%%.....
// .%%%%%...%%%%%...%%..%%..%%%%...
// .%%..%%..%%..%%..%%..%%..%%.....
// .%%%%%...%%..%%..%%%%%...%%.....
// ................................
// 

const float MAT_PBR = 0.0;
const float MAT_UNLIT = 1.0;

// vec4 at->vec3 albedo, float type
// vec2 mr->vec2 metallic, roughness

float pow5(float x)
{
    return (x * x) * (x * x) * x;
}

vec3 Microfacet_BRDF(vec3 albedo, float metallic, float paramRoughness, vec3 L, vec3 V, vec3 N, bool isSecondary)
{
    // i think 0.5
    const float reflectance = 0.5;

    float roughness = paramRoughness * paramRoughness;
    // clamp roughness
    roughness = max(roughness, 1e-3);
    vec3 f0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + albedo * metallic;

    vec3 H = normalize(L + V);
    float NoV = saturate(dot(N, V)) + 1e-5;
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));
    float LoH = saturate(dot(L, H));

    // Calc specular
    // Calc D
    float da = NoH * roughness;
    float dk = roughness / (1.0 - NoH * NoH + da * da);
    float D_spec = dk * dk / PI;
    // Calc V
    float va = roughness * roughness;
    float G_V = NoL * sqrt(max(0.0, NoV * NoV * (1.0 - va) + va));
    float G_L = NoV * sqrt(max(0.0, NoL * NoL * (1.0 - va) + va));
    float V_spec = 0.5 / (G_V + G_L);
    // Calc F
    // vec3 F_spec = f0 + (1.0 - f0) * pow(1.0 - LoH, 5.0);
    vec3 F_spec = f0 + (1.0 - f0) * pow5(1.0 - LoH);
    vec3 Fr = (isSecondary ? 1.0 : D_spec * V_spec) * F_spec;

    // Calc diffuse
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float ls = 1.0 + (f90 - 1.0) * pow5(1.0 - NoL);
    float vs = 1.0 + (f90 - 1.0) * pow5(1.0 - NoV);
    float fdb = ls * vs / PI;
    vec3 diffColor = albedo * (1.0 - metallic);
    vec3 Fd = diffColor * fdb;

    return (Fr + Fd) * NoL;
}

// 
// ..%%%%...%%%%%...%%%%%%.
// .%%......%%..%%..%%.....
// ..%%%%...%%..%%..%%%%...
// .....%%..%%..%%..%%.....
// ..%%%%...%%%%%...%%.....
// ........................
// 

// https://mercury.sexy/hg_sdf/
float fOpUnionRound(float a, float b, float r)
{
    vec2 u = max(vec2(r - a, r - b), vec2(0));
    return max(r, min(a, b)) - length(u);
}
float fOpDifferenceRound(float a, float b, float r)
{
    vec2 u = max(vec2(r + a, r - b), vec2(0));
    return min(-r, max(a, -b)) + length(u);
}

float sdBox(vec3 p, vec3 b)
{
    vec3 d = abs(p) - b * 0.5;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

const vec2 RoomSize = vec2(6, 4) * 0.5;
vec2 opRoomRep(inout vec3 p)
{
    vec2 ip = floor(p.xy / RoomSize) * RoomSize + RoomSize * 0.5;
    p.xy = mod(p.xy, RoomSize) - 0.5 * RoomSize;
    return ip;
}

float sdf(vec3 p)
{
    vec3 op = p;
    vec2 ip = opRoomRep(p);
    vec3 h3 = pcg33(vec3(ip, 0));

    float td, td2;
    vec3 tp;

    bool isfloorB = fract(0.5 * ip.x / RoomSize.x) < 0.5;
    float isfloor = float(!isfloorB) * 1e9;
    float isroom = float(isfloorB) * 1e9;

    // かいだん
    // みぞ
    const float w = 0.04;
    tp = abs(fract(op * 10.0) - 0.5);
    float mizo = dot(vec3(1), smoothstep(w, w * 0.5, tp)) / 3.0 * 0.001;
    // かいだん
    const float slope = RoomSize.x * 0.2;
    const float bd = slope * 2.0 / sqrt(RoomSize.y * RoomSize.y + slope * slope * 4.0);// bound
    float cy = linearstep(-RoomSize.x * 0.5 + slope, RoomSize.x * 0.5 - slope, p.x) * RoomSize.y + RoomSize.y * 0.25;
    float fy = p.y - cy;
    float fd = min(min(max((min(abs(fy), abs(fy + RoomSize.y)) - RoomSize.y * 0.25) * bd, -p.z), -p.z + 3.0), max(abs(p.x) - RoomSize.x * 0.5 + slope, -p.z + 0.5));
    fd += mizo;
    // パイプ
    tp = p - vec3(-RoomSize.x * 0.45, 0, -0.07);
    tp.x = abs(tp.x - 0.1) - 0.1;
    fd = min(fd, length(tp.xz) - 0.07);
    // パイプの金具
    tp.y = abs(fract(tp.y) - 0.5) - 0.06;
    // sdBox(p-vec3(0,h*0.5,0),vec3(0,h,0))-r;
    // fd = min(fd, sdVerticalCapsule((tp - vec3(-0.11, 0, 0)).yxz, 0.04, 0.01));
    fd = min(fd, sdBox(tp+vec3(0.1, 0, 0),vec3(0.05,0,0))-0.01);
    vec2 dd = abs(vec2(length(tp.xz), tp.y)) - vec2(0.08, 0.02);
    fd = min(fd, min(max(dd.x, dd.y), 0.0) + length(max(dd, 0.0)));
    // 階段限定
    fd += isfloor;

    // 部屋
    // ベースの壁
    float hd = -p.z;
    // 部屋のあな
    hd = fOpDifferenceRound(hd, sdBox(p, vec3(RoomSize - 0.2, 4.0)), 0.01);
    // 窓枠
    tp = p - vec3(0, 0, 0.75);
    tp.x = abs(tp.x);
    const vec2 rsize = RoomSize - 0.2;
    hd = fOpUnionRound(hd, sdBox(tp, vec3(rsize, 0.05)), 0.01);
    const vec3 msize = vec3(rsize.x * 0.25 - 0.125, rsize.y * 0.75, 0.1);
    hd = fOpDifferenceRound(hd, sdBox(tp - vec3(msize.x * 0.5 + 0.05, -msize.y * 0.1, 0), msize), 0.01);
    const vec3 msize2 = vec3(msize.x, msize.y * 0.5, msize.z);
    hd = fOpDifferenceRound(hd, sdBox(tp - vec3(msize2.x * 1.5 + 0.15, msize2.y * 0.3, 0), msize2), 0.01);
    // 腰壁
    hd = min(hd, -0.01 + sdBox(p - vec3(0, -RoomSize.y * 0.25, 0.1), vec3(RoomSize.x - 0.2, RoomSize.y * 0.3, 0.025)));
    // 腰壁の穴
    tp = p - vec3(0, -RoomSize.y * 0.25, 0.1);
    tp.x = opRepLim(tp.x, 0.2, 6);
    hd = fOpDifferenceRound(hd, sdBox(tp, vec3(0.06, RoomSize.y * 0.25, 0.05)), 0.01);
    // 手すり
    hd = min(hd, -0.01 + sdBox(p - vec3(0, -RoomSize.y * 0.1 - 0.025, 0), vec3(RoomSize.x - 0.2, 0.05, 0.2)));

    // 室外機
    const vec3 size = vec3(0.4, 0.28, 0.1);
    td = -0.01 + sdBox(p - vec3(1.1, -0.77, -0.02), size);
    tp = p - vec3(1.1, -0.77, -0.02) - vec3(-0.06, 0, -0.1);
    td = fOpDifferenceRound(td, sdBox(tp,vec3(0,0,0.1))-0.1, 0.005);
    tp.z -= 0.05;
    td = min(td, length(tp)-0.015);
    // ひだひだ
    const float div = PI / 32.;
    float a = atan(tp.y, tp.x);
    a = mod(a, div * 2.0) - div;
    tp.xy = orbit(a) * length(tp.xy);
    td = fOpUnionRound(td, sdBox(tp,vec3(0.2,0,0))-0.002, 0.005);
    hd = min(hd, td);
    // 部屋限定
    hd += isroom;

    // 部屋と階段の合成
    float d = min(hd, fd);

    // カーテン
    float shimaru = smoothstep(0.0, 1.5, 0.9 - p.y);
    const vec3 ksize = vec3(RoomSize - 0.4, 0.01);
    vec3 kp = p;
    kp.x *= mix(1.1, 1.0, shimaru);
    float shf = h3.x * TAU;
    float lt = Time * 0.5 + shf;
    float kz = cos(shf + kp.x * PI * 7.0 + 0.5 * PI * cos(kp.x * 7.0)) * 0.05 + cos(kp.x * 2.0 + lt * 2.0 + 0.5 * PI * cos(lt)) * 0.05;
    kp -= vec3(0, 0, 1.0 + kz * shimaru);
    td = sdBox(kp, ksize) * 0.5 - 0.01;
    // 部屋限定
    td += isroom;
    // カーテンを合成
    MatID = (td < d ? 1 : 0);
    d = min(d, td);

    // 部屋のライト
    td2 = length(p - vec3(sign(h3.x - 0.5), -0.09, 0)) - 0.1;
    // 部屋限定
    td2 += isroom;
    // 階段のライト
    td = length(p - vec3(-RoomSize * 0.5 + 0.1, 2.9)) - 0.1;
    // 階段限定
    td += isfloor;
    // 部屋と階段のライトを合成
    td = min(td, td2);
    // ライトを合成
    MatID = (td < d ? 2 : MatID);
    d = min(d, td);

    return d;
}

// ishit,shadow
vec2 march(vec3 rd, vec3 ro, out vec3 rp)
{
    const float w = 0.02;
    const float minv = 0.1;
    float v = 1.0, ph = LenMax;
    float dist, len = 0.0;

    for(int i = 0; i < LoopMax; i++)
    {
        rp = ro + rd * len;

        float _lt;
        // abs z
        if(tl(90.0, 105.0, _lt))
        {
            rp.z = abs(rp.z) - 2.0;
        }
        // polar
        if(tl(105.0, 120.0, _lt))
        {
            rp.xz = vec2((atan(rp.z, rp.x) + PI) / TAU * RoomSize.x * 8.0, length(rp.xz));
            rp.z -= 4.0;
        }

        dist = sdf(rp);

        // shadow
        float y = dist * dist / (2.0 * ph);
        float d = sqrt(dist * dist - y * y);
        v = min(v, d / (w * max(0.0, len - y)));
        ph = dist;

        // traverse
        vec2 irp = (floor((rp.xy + sign(rd.xy) * RoomSize) / RoomSize) + 0.5) * RoomSize;
        vec2 bd = abs(irp - rp.xy) - 0.5 * RoomSize;
        bd = max(bd, 0.0) / abs(rd.xy) + DistMin + float(rp.z < -2.0) * LenMax;
        dist = min(dist, min(bd.x, bd.y));

        len += dist;
        if(dist < DistMin)
        {
            return vec2(1, minv);
        }
        if(len > LenMax)
        {
            return vec2(0, max(v, minv));
        }
    }
    // トラバーサルのせいでlenがLenMaxを越えないことがある
    return vec2(0, max(v, minv));
}

// 
// ..%%%%...%%..%%...%%%%...%%%%%...%%%%%%..%%..%%...%%%%..
// .%%......%%..%%..%%..%%..%%..%%....%%....%%%.%%..%%.....
// ..%%%%...%%%%%%..%%%%%%..%%..%%....%%....%%.%%%..%%.%%%.
// .....%%..%%..%%..%%..%%..%%..%%....%%....%%..%%..%%..%%.
// ..%%%%...%%..%%..%%..%%..%%%%%...%%%%%%..%%..%%...%%%%..
// ........................................................
// 

const vec3 DirectionalLight = normalize(vec3(1, 1, -1));

vec3 sky(vec3 rd)
{
    vec3 gcol = vec3(0.01);
    float theta = atan(rd.x, -rd.z);
    float phi = acos(rd.y);
    vec2 th = vec2(theta, phi * 2.0) / TAU * 20.0 + vec2(0.05, 0.01) * Time;
    vec3 scol = vec3(mix(0.1, 0.5, fbm32(th).x));
    float up = saturate((rd.y + 0.5) * 0.5);
    vec3 col = mix(gcol, scol, up);
    // sun
    float sun = saturate(dot(rd, DirectionalLight));
    sun = smoothstep(0.8, 1.0, sun);
    col *= mix(1.0, 3.0, sun);
    col = mix(col, vec3(1.0), pow(linearstep(0.995, 1.0, sun * sun), 5.0));
    col *= linearstep(1.0, 0.9, rd.y);
    return col;
}

vec3 shading(inout vec3 P, vec3 V, vec3 N)
{
    // triplanar
    vec3 tN = sign(N) * abs(N);
    tN = tN / dot(vec3(1), tN);
    vec2 uv = tN.x * P.zy + tN.y * P.xz + tN.z * P.xy;

    // identify
    vec3 RP = P;
    vec2 ip = opRoomRep(RP);
    bool isfloorB = fract(0.5 * ip.x / RoomSize.x) < 0.5;
    vec3 h3 = pcg33(vec3(0.1, ip));

    // get mat
    float type = MAT_PBR;
    vec3 albedo = vec3(1);
    float roughness = 0.5;
    float metallic = 0.5;
    // ポイントライトの色
    vec3 plcol = (step(h3.y, 0.9) * 0.8 + 0.2) * mix(k2000, k12000, h3.x);
    if(MatID == 0)
    {
        // Mat:Concrete
        vec3 fbm = fbm32(uv * 3.0 * vec2(3, 1));// gravity ydown
        vec3 fbm2 = fbm32(uv * 96.0 * vec2(2, 1));
        albedo = vec3(saturate(1.3 * mix(0.6, 1.0, fbm.y) * mix(0.8, 1.0, pow(fbm2.x, 3.0))));
        roughness = mix(0.5, 1.0, pow(fbm.y, 3.0));
        metallic = 0.01;
        N = normalize(N + (fbm * 2.0 - 1.0) * 0.05);
    }
    else if(MatID == 1)
    {
        // Mat:カーテン
        float h = h3.x;
        albedo = (h < 0.7 ? vec3(0.8, 0.7, 0.6) : (h < 0.8 ? vec3(0.8, 0.2, 0.2) : (h < 0.9 ? vec3(0.8, 0.6, 0.3) : vec3(0.5, 0.7, 0.8))));
        roughness = 0.99;
        metallic = 0.01;
    }
    else
    {
        // Mat:ライト
        type = MAT_UNLIT;
        albedo = plcol;
    }

    // avoid self-intersection
    P += N * DistMin * 2.0;
    // directional shadow
    vec3 _rp;
    vec2 sh = march(DirectionalLight, P, _rp);
    float visible = sh.y;

    // primary shading
    vec3 col = vec3(0);
    // directional light
    col += visible * Microfacet_BRDF(albedo, metallic, roughness, DirectionalLight, V, N, false) * k12000;
    // point light
    vec3 L = (isfloorB ? vec3(-RoomSize * 0.5 + 0.1, 2.9) : vec3(sign(h3.x - 0.5), -0.06, 0)) - RP;
    float l = length(L);
    L /= l;
    float d = max(0.0, l);
    vec3 lcol = plcol * pow(1.0 / (1.0 + d), 2.0);
    col += Microfacet_BRDF(albedo, metallic, roughness, L, V, N, false) * lcol;

    // ao
    col *= sqrt(saturate(sdf(P + N * 0.05) / 0.05));
    return mix(col, albedo, type);
}

// 
// .%%%%%....%%%%....%%%%...%%%%%%.
// .%%..%%..%%..%%..%%..%%....%%...
// .%%%%%...%%..%%..%%..%%....%%...
// .%%..%%..%%..%%..%%..%%....%%...
// .%%..%%...%%%%....%%%%.....%%...
// ................................
// 

const vec2 FC = gl_FragCoord.xy;
const vec2 RES = resolution.xy;
const vec2 ASP = RES / min(RES.x, RES.y);
vec2 FocalUV = vec2(0.5);

void getRORD(out vec3 ro, out vec3 rd, out vec3 dir, vec2 suv)
{
    // Parameter
    float fov = 60.0;
    float fisheye = 0.0;

    ro = vec3(1);
    dir = vec3(0, 0, 1);
    float lt = 0.0;
    // Time = 46.28;
    if(tl(0.0, 15.0, lt))
    {
        ro = vec3(0.2, 0.2, mix(2.2, 1.5, lt));
        dir = normalize(mix(vec3(0.5, 1.5, 1), vec3(0.1, 0.1, 1), lt));
    }
    else if(tl(15.0, 30.0, lt))
    {
        fov = mix(60.0, 90.0, lt);
        fisheye = mix(0.0, 0.4, lt);
        ro = mix(vec3(0.4, 0.2, 2.0), vec3(0.35, 0.6, 0.4), lt);
        dir = normalize(mix(vec3(-0.5, -1, -1), vec3(0, 0, -1), lt));
    }
    else if(tl(30.0, 45.0, lt))
    {
        ro = vec3(mix(0.8 - RoomSize.x, -0.1, lt), -1.0, 0);
        dir = normalize(vec3(-1, -0.1, 0.3));
    }
    else if(tl(45.0, 60.0, lt))
    {
        ro = vec3(mix(-0.5, 0.2 - RoomSize.x, lt), -1.8, -0.3);
        dir = normalize(vec3(1, 0.2, 0.8));
    }
    else if(tl(60.0, 75.0, lt))
    {
        float z = floor(lt * 3.0) * 8.0 + fract(lt * 3.0);
        ro = vec3(0, 0, -1.0 - z);
        dir = vec3(0, 0, 1);
    }
    else if(tl(77.0, 90.0, lt))
    {
        float t = 20.0 * lt;
        vec3 ta = vec3(-RoomSize.x * 0.5 + 0.3, t * RoomSize.y, 0.0);
        ro = ta + vec3(mix(3, -3, lt), -5, -3);
        dir = normalize(ta - ro);
    }
    else if(tl(90.0, 105.0, lt))
    {
        ro = vec3(-Time * 4.0, 0, 0);
        dir = normalize(vec3(-1, mix(0.2, -0.2, lt), 0));
    }
    else if(tl(105.0, 120.0, lt))
    {
        ro = vec3(0, Time * 2.0, 0);
        dir = normalize(vec3(0, 1, mix(0.0, 1.0, lt)));
        dir.xz *= rot(PI * lt);
    }

    vec3 tebure = (fbm32(vec2(-4.2, Time * 0.1)) * 2.0 - 1.0) * 0.05;
    ro += tebure;

    vec3 N = vec3(0, 1, 0);
    vec3 B = normalize(cross(N, dir));
    N = normalize(cross(dir, B));
    mat3 bnt = mat3(B, N, dir);
    float zf = 1.0 / tan(fov * PI / 360.0);
    zf -= zf * length(suv) * fisheye;

    rd = normalize(bnt * vec3(suv, zf));
}

vec4 tracer(vec3 rd, vec3 ro)
{
    vec3 rp;
    if(march(rd, ro, rp).x < 0.5)
    {
        return vec4(sky(rd), LenMax);
    }
    float depth = length(rp - ro);
    const float h = NormalEPS;
    const vec2 k = vec2(1, -1);
    int mid = MatID;
    vec3 N = normalize(k.xyy * sdf(rp + k.xyy * h) + k.yyx * sdf(rp + k.yyx * h) + k.yxy * sdf(rp + k.yxy * h) + k.xxx * sdf(rp + k.xxx * h));
    MatID = mid;
    vec3 col = shading(rp, -rd, N);
    return vec4(col, depth);
}

// 
// .%%...%%...%%%%...%%%%%%..%%..%%.
// .%%%.%%%..%%..%%....%%....%%%.%%.
// .%%.%.%%..%%%%%%....%%....%%.%%%.
// .%%...%%..%%..%%....%%....%%..%%.
// .%%...%%..%%..%%..%%%%%%..%%..%%.
// .................................
// 

float getcoc(float d, float focalPlane)
{
    // focalLength:焦点距離
    // focalPlane:ピントを合わせたいカメラまでの距離
    const float focalLength = 0.3;
    // const float focalLength = 0.35;
    const float cocScale = 10.0;
    // 開口径
    float aperture = min(1.0, focalPlane * focalPlane);
    float ca = aperture * focalLength * (focalPlane - d);
    float cb = abs(d * (focalPlane - focalLength)) + 1e-9;
    float c = abs(ca / cb) * cocScale;
    return c;
}

vec3 acesFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

vec3 postprocess(vec3 col, vec3 seed)
{
    // noise乗せた方が雰囲気いいか？
    col += pcg33(seed) * 0.05;
    // 嘘 Gamma Correction
    col = pow(col, vec3(0.8));
    // col = pow(col, vec3(0.4545));
    col = acesFilm(col);
    // カラグレ
    #define COG(_s,_b) col._s = smoothstep(0.0-(_b)*0.5,1.0+(_b)*0.5,col._s)
    COG(r, 0.05);
    // COG(g, -0.1);
    // COG(b, 0.05);

    float lt = 0.0;
    if(tl(0.0, 10.0, lt))
    {
        // 開幕
        col *= lt * lt;
    }
    else if(tl(14.0, 16.0, lt))
    {
        col *= 2.0 * abs(lt - 0.5);
    }
    else if(tl(29.0, 31.0, lt))
    {
        col *= 2.0 * abs(lt - 0.5);
    }
    else if(tl(44.0, 46.0, lt))
    {
        col *= 2.0 * abs(lt - 0.5);
    }
    else if(tl(59.0, 61.0, lt))
    {
        col *= 2.0 * abs(lt - 0.5);
    }
    else if(tl(74.0, 77.0, lt))
    {
        col *= saturate(75.0 - Time);
    }
    else if(tl(119.0, 120.0, lt))
    {
        col *= 1.0 - lt;
    }
    else if(Time >= 120.0)
    {
        col *= 0.0;
    }

    return col;
}

void main()
{
    // Set Time
    Time = time;
    // Get UVs
    vec2 uv = FC / RES;
    // TAA
    uv += (pcg33(vec3(FC, Time)).xy - 0.5) * 0.25 / RES;
    vec2 suv = (uv - 0.5) * 2.0 * ASP;

    // Camera
    vec3 ro, rd, dir;
    getRORD(ro, rd, dir, suv);

    // Trace
    vec4 traced = tracer(rd, ro);
    // Color Grading
    traced.rgb = postprocess(traced.rgb, vec3(-Time, 4.2 * uv));
    // TAA&MotionBlur
    const float ema = 0.5;
    vec4 back = texture(backBuffer0, uv);
    outColor0 = mix(traced, back, ema);

    /*
    outColor1 = max(vec4(0), traced);

    // DOF&Bloom
    vec4 col = vec4(0);
    float focalPlane = textureLod(backBuffer1, FocalUV, 3.0).a;
    vec4 uvtex = texture(backBuffer1, uv);
    float depth = uvtex.a;
    float coc = getcoc(depth, focalPlane);
    const int N = 64;
    const float SN = sqrt(float(N));

    for(int i = 0; i < N; i++)
    {
        float fi = float(i);
        float r = coc * sqrt(fi) / SN;
        float th = fi * GOLD;
        vec2 suv = uv + orbit(th) * r / resolution;

        vec4 stex = texture(backBuffer1, suv);
        float sd = stex.a;

        if(sd > 0.0)
        {
            float scoc = getcoc(sd, focalPlane);
            float w = max(1e-3, scoc);
            col += vec4(stex.rgb, 1) * w;
        }
    }
    col.rgb /= col.a;

    // TAA&MotionBlur
    const float ema = 0.5;
    vec4 back = texture(backBuffer0, uv);
    vec3 res = mix(col.rgb, back.rgb, ema);
    res = max(vec3(0), res);
    outColor0 = vec4(res, 1);
    */
}
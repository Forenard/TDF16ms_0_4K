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

// out vec4 outColor0;
layout(location = 0) out vec4 outColor0;
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
const float TAU = PI * 2.0;
const float GOLD = PI * (3.0 - sqrt(5.0));// 2.39996...
const vec3 k2000 = vec3(1, 0.5, 0.1);
const vec3 k12000 = vec3(0.8, 0.8, 0.9);
const float STEP2TIME = 60.0 / 128.0 / 4.0;

#define saturate(x) clamp(x,0.0,1.0)

vec2 orbit(float t)
{
    return vec2(cos(t), sin(t));
}

mat2 rot(float x)
{
    vec2 v = orbit(x);
    return mat2(v.x, v.y, -v.y, v.x);
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

vec3 Microfacet_BRDF(vec3 albedo, float metallic, float paramRoughness, vec3 L, vec3 V, vec3 N)
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
    vec3 Fr = D_spec * V_spec * F_spec;

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

float sdBox(vec3 p, vec3 b)
{
    vec3 d = abs(p) - b * 0.5;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

vec2 IP;
float sdf(vec3 p)
{
    IP = floor(p.xy / vec2(3, 2)) * vec2(3, 2) + vec2(1.5, 1);
    vec3 op = p;
    p.xy -= IP;
    vec3 h3 = pcg33(vec3(IP, 1));

    float td, td2;
    vec3 tp;

    bool isfloorB = fract(0.5 * IP.x / 3.0) < 0.5;
    float isroom = float(isfloorB) * 1e9;

    // かいだん
    // みぞ
    tp = abs(fract(op * 10) - 0.5);
    float mizo = min(dot(vec3(1), smoothstep(0.02, 0.0, tp)), 1) * 0.001;
    // かいだん
    // #define linearstep(a, b, x) min(max((p.x+1) / ((b) - (a)), 0.0), 1.0)
    float fy = p.y - saturate((p.x + 1) * 0.5) * 2 - 0.5;
    float fd = min(min(max((min(abs(fy), abs(fy + 2)) - 0.5) * 0.4, -p.z), -p.z + 3), max(abs(p.x) - 1, -p.z + 0.5));
    fd += mizo;
    // パイプ
    tp = p - vec3(-1.35, 0, -0.07);
    tp.x = abs(tp.x - 0.1) - 0.1;
    fd = min(fd, length(tp.xz) - 0.07);
    // パイプの金具
    tp.y = abs(fract(tp.y) - 0.5) - 0.07;
    fd = min(fd, sdBox(tp + vec3(0.1, 0, 0), vec3(0.05, 0, 0)) - 0.01);
    vec2 dd = abs(vec2(length(tp.xz), tp.y)) - vec2(0.08, 0.02);
    fd = min(fd, min(max(dd.x, dd.y), 0.0) + length(max(dd, 0.0)));

    // 部屋
    // ベースの壁
    float hd = -p.z - 0.11;
    // 部屋のあな
    hd = max(hd, -sdBox(p, vec3(2.8, 1.8, 4)));
    // 窓枠
    tp = p;
    tp.z -= 0.75;
    tp.x = abs(abs(tp.x) - 0.4);
    hd = min(hd, sdBox(tp, vec3(3, 2, 0.05)));
    hd = max(hd, -sdBox(tp - vec3(0.4, -0.135, 0), vec3(0.7, 1.35, 0.1)));
    // 腰壁
    hd = min(hd, sdBox(p - vec3(0, -0.5, 0.1), vec3(2.8, 0.6, 0.025)) - 0.01);
    // 腰壁の穴
    tp = p - vec3(0.2 * clamp(round(p.x / 0.2), -6, 6), -0.5, 0.1);
    hd = max(hd, -sdBox(tp, vec3(0.1, 0.5, 0.05)));
    // 手すり
    hd = min(hd, sdBox(p + vec3(0, 0.225, 0), vec3(2.8, 0.05, 0.2)) - 0.01);

    // 室外機
    tp = p - vec3(1.15, -0.76, 0);
    td = sdBox(tp, vec3(0.4, 0.3, 0.15) - 0.02);
    tp = tp - vec3(-0.06, 0, -0.11);
    td = max(td, -sdBox(tp, vec3(0, 0, 0.1)) + 0.1);
    // 室外機の柵
    tp.z -= 0.05;
    tp.y -= 0.01 * clamp(round(tp.y / 0.01), -9, 9);
    td = min(td, sdBox(tp, vec3(0.2, 0, 0)) - 0.002);
    hd = min(hd, td);

    // 部屋と階段の合成
    float d = isfloorB ? fd : hd;

    // カーテン
    float shimaru = smoothstep(0.0, 2.0, 1.0 - p.y);
    vec3 kp = p;
    kp.x *= mix(1.2, 1.0, shimaru);
    // float shf = h3.x * TAU;
    float lt = Time + h3.x * TAU;
    float kz = cos(kp.x * PI * 8.0 + cos(kp.x * 8.0)) * 0.05 + sin(kp.x * 4.0 + lt + 0.5 * cos(lt)) * 0.05;
    kp.z -= 1.0 + kz * shimaru;
    td = sdBox(kp, vec3(2.6, 1.6, 0.01)) * 0.5 - 0.01;
    // 部屋限定
    td += isroom;
    // カーテンを合成
    MatID = (td < d ? 1 : 0);
    d = min(d, td);

    // 部屋のライト
    td2 = length(p - vec3(sign(h3.x - 0.5), -0.09, 0)) - 0.1;
    // 階段のライト
    td = length(p - vec3(-1.4, -0.9, 2.9)) - 0.1;
    // 部屋と階段のライトを合成
    td = isfloorB ? td : td2;
    // ライトを合成
    MatID = (td < d ? 2 : MatID);
    d = min(d, td);

    return d;
}

// ishit,shadow
vec2 march(vec3 rd, vec3 ro, out vec3 rp)
{
    const float w = 0.01;
    const float minv = 0.01;
    float v = 1.0, ph = LenMax;
    float dist, len = 0.0;
    for(int i = 0; i < LoopMax; i++)
    {
        rp = ro + rd * len;

        // abs z
        rp.z = (90.0 <= Time && Time < 105.0 ? abs(rp.z) - 2.0 : rp.z);
        // polar
        rp.xz = (105.0 <= Time && Time < 120.0 ? vec2((atan(rp.z, rp.x) + PI) / TAU * 3 * 16, length(rp.xz) - 4.0) : rp.xz);
        // beat shift
        // float bt = Time / STEP2TIME / 8;
        // const float span = 4.0;
        // float sy = floor(bt / span) + smoothstep(0.0,1.0,pow(min(1, mod(bt, span)), 0.3));
        // bt += span * 0.5;
        // float sx = floor(bt / span) + smoothstep(0.0,1.0,pow(min(1, mod(bt, span)), 0.3));
        // rp.y += 2.0 * sy * sign(fract(rp.x / 6) - 0.5);
        // rp.x += 3.0 * sx * sign(fract(rp.y / 4) - 0.5);

        const float span = 4.0;
        float bt = time / STEP2TIME / 4 + floor(rp.x / 6);
        float sy = floor(bt / span) - pow(1.0 / (1.0 + mod(bt, span)), 5.0);
        rp.y += 3.0 * sy * sign(fract(rp.x / 12) - 0.5) * step(75.0, Time);
        // rp.y += 2.0 * time;

        // sdf
        dist = sdf(rp);

        // shadow
        float y = dist * dist / (2.0 * ph);
        float d = sqrt(dist * dist - y * y);
        v = min(v, d / (w * max(0.0, len - y)));
        ph = dist;

        // traverse
        vec2 irp = floor(rp.xy / vec2(3, 2) + sign(rd.xy)) * vec2(3, 2) + vec2(1.5, 1);
        vec2 bd = abs(irp - rp.xy) - vec2(1.5, 1);
        bd = max(bd, 0.0) / abs(rd.xy) + DistMin + step(rp.z, -3.0) * LenMax;
        float td = min(bd.x, bd.y);

        // sdf
        dist = min(td, dist);
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
    return vec2(1, max(v, minv));
}

// 
// ..%%%%...%%..%%...%%%%...%%%%%...%%%%%%..%%..%%...%%%%..
// .%%......%%..%%..%%..%%..%%..%%....%%....%%%.%%..%%.....
// ..%%%%...%%%%%%..%%%%%%..%%..%%....%%....%%.%%%..%%.%%%.
// .....%%..%%..%%..%%..%%..%%..%%....%%....%%..%%..%%..%%.
// ..%%%%...%%..%%..%%..%%..%%%%%...%%%%%%..%%..%%...%%%%..
// ........................................................
// 

const vec3 DirectionalLight = normalize(vec3(1, 1.5, -1));

// 
// .%%...%%...%%%%...%%%%%%..%%..%%.
// .%%%.%%%..%%..%%....%%....%%%.%%.
// .%%.%.%%..%%%%%%....%%....%%.%%%.
// .%%...%%..%%..%%....%%....%%..%%.
// .%%...%%..%%..%%..%%%%%%..%%..%%.
// .................................
// 

void main()
{
    // 
    // ..%%%%...%%%%%%..%%%%%%..%%%%%%..%%%%%%..%%..%%...%%%%..
    // .%%......%%........%%......%%......%%....%%%.%%..%%.....
    // ..%%%%...%%%%......%%......%%......%%....%%.%%%..%%.%%%.
    // .....%%..%%........%%......%%......%%....%%..%%..%%..%%.
    // ..%%%%...%%%%%%....%%......%%....%%%%%%..%%..%%...%%%%..
    // ........................................................
    // 
    // Set Time
    // const float bar = 120.0/32.0;
    // Time = (floor(time/bar)+pow(fract(time/bar),0.5))*bar;
    Time = time;
    // Time = 75.0 + time;
    // Get UVs
    const vec2 fc = gl_FragCoord.xy, res = resolution.xy, asp = res / min(res.x, res.y);
    const vec2 uv = fc / res;
    // TAA
    const vec3 h3 = pcg33(vec3(fc, Time));
    const vec2 suv = (uv - 0.5 + ((h3.xy - 0.5) * 0.25 / res)) * 2.0 * asp;
    // const vec2 suv = (uv - 0.5) * 2.0 * asp;

    // 
    // ..%%%%....%%%%...%%...%%..%%%%%%..%%%%%....%%%%..
    // .%%..%%..%%..%%..%%%.%%%..%%......%%..%%..%%..%%.
    // .%%......%%%%%%..%%.%.%%..%%%%....%%%%%...%%%%%%.
    // .%%..%%..%%..%%..%%...%%..%%......%%..%%..%%..%%.
    // ..%%%%...%%..%%..%%...%%..%%%%%%..%%..%%..%%..%%.
    // .................................................
    // 
    // Parameter
    vec3 ro = vec3(1), dir = vec3(0, 0, -1), rd;
    // シーケンス
    const vec3 ro0[] = vec3[](vec3(0.2, 0.2, 2.2), vec3(0.4, 0.2, 2.0), vec3(-2.2, -1.0, 0), vec3(-0.5, -1.8, -0.3), vec3(0, 0, -3), vec3(0, 0, -6), vec3(0), vec3(0));
    const vec3 ro1[] = vec3[](vec3(0.1, 0.1, 1.5), vec3(0.35, 0.6, 0.4), vec3(-0.1, -1.0, 0), vec3(-2.8, -1.8, -0.3), vec3(0, 0, -15), vec3(0, 30, -9), vec3(-70, 0, 0), vec3(0, 70, 0));
    const vec3 dir0[] = vec3[](vec3(0.5, 1.5, 1), vec3(-0.5, -1, -1), vec3(-1, -0.1, 0.3), vec3(0.8, 0.2, 1), vec3(0, 0, 1), vec3(-0.2, 0, 1), vec3(-1, 0.2, 0), vec3(1, 1, 0));
    const vec3 dir1[] = vec3[](vec3(0.1, 0.1, 1), vec3(0, 0, -1), vec3(-1, -0.1, 1), vec3(0.8, 0.2, 1.5), vec3(0, 0, 1), vec3(0.2, 0, 1), vec3(-1, -0.2, 0), vec3(0, 1, 0.1));
    float ft = Time / 15.0;
    int id = int(ft) % 8;
    float lt = fract(ft) * step(ft, 8);
    float mx = (id == 4 ? (floor(lt * 4.0) + lt) * 0.25 : lt);
    ro = mix(ro0[id], ro1[id], mx);
    dir = normalize(mix(dir0[id], dir1[id], mx));
    // ADD 手振れ
    // const vec3 tebure = (fbm32(vec2(2, Time * 0.1)) - 0.5) * 0.16;
    // ro += tebure;
    // rdを計算
    const vec3 B = normalize(cross(vec3(0, 1, 0), dir));
    const mat3 bnt = mat3(B, normalize(cross(dir, B)), dir);
    const float fov = 75.0;
    rd = normalize(mat3(B, normalize(cross(dir, B)), dir) * vec3(suv, 1.0 / tan(fov * PI / 360.0)));

    // 
    // .%%%%%....%%%%...%%..%%..........%%%%%%..%%%%%....%%%%....%%%%...%%%%%%.
    // .%%..%%..%%..%%...%%%%.............%%....%%..%%..%%..%%..%%..%%..%%.....
    // .%%%%%...%%%%%%....%%..............%%....%%%%%...%%%%%%..%%......%%%%...
    // .%%..%%..%%..%%....%%..............%%....%%..%%..%%..%%..%%..%%..%%.....
    // .%%..%%..%%..%%....%%..............%%....%%..%%..%%..%%...%%%%...%%%%%%.
    // ........................................................................
    // 
    // Trace
    vec3 P;
    vec2 hit = march(rd, ro, P);
    // float depth = min(length(P - ro),LenMax);
    // get normal
    float h = NormalEPS;
    vec2 k = vec2(1, -1);
    int mid = MatID;
    vec3 N = normalize(k.xyy * sdf(P + k.xyy * h) + k.yyx * sdf(P + k.yyx * h) + k.yxy * sdf(P + k.yxy * h) + k.xxx * sdf(P + k.xxx * h));
    MatID = mid;

    // 
    // ..%%%%...%%..%%...%%%%...%%%%%...%%%%%%..%%..%%...%%%%..
    // .%%......%%..%%..%%..%%..%%..%%....%%....%%%.%%..%%.....
    // ..%%%%...%%%%%%..%%%%%%..%%..%%....%%....%%.%%%..%%.%%%.
    // .....%%..%%..%%..%%..%%..%%..%%....%%....%%..%%..%%..%%.
    // ..%%%%...%%..%%..%%..%%..%%%%%...%%%%%%..%%..%%...%%%%..
    // ........................................................
    // 
    // triplanar
    vec3 tN = sign(N) * abs(N);
    tN = tN / dot(vec3(1), tN);
    vec2 tuv = (tN.x * P.zy + tN.y * P.xz + tN.z * P.xy) * vec2(3, 1);
    // identify
    vec3 RP = P;
    RP.xy -= IP;
    vec3 hash = pcg33(vec3(IP, 1));
    bool isfloorB = fract(0.5 * IP.x / 3.0) < 0.5;
    // get mat
    float type = MAT_PBR;
    vec3 albedo = vec3(1);
    float roughness = 0.5;
    float metallic = 0.5;
    // ポイントライトの色
    vec3 plcol = mix(k2000, k12000, hash.x);
    if(MatID == 0)
    {
        // Mat:Concrete
        vec3 fbm = fbm32(tuv * 3.0);// gravity ydown
        vec3 fbm2 = fbm32(tuv * 96.0);
        albedo = vec3(min(1.3 * mix(0.5, 1.0, fbm.y) * mix(0.9, 1.0, pow(fbm2.x, 3.0)), 1));
        // roughness = mix(0.5, 1.0, pow(fbm.y, 3.0));
        roughness = albedo.r * 0.5;
        metallic = 0.01;
        N = normalize(N + (fbm - 0.5) * 0.1);
    }
    else if(MatID == 1)
    {
        // Mat:カーテン
        float h = hash.x;
        const vec3 cols[] = vec3[](vec3(0.8, 0.7, 0.6), vec3(0.8, 0.2, 0.2), vec3(0.5, 0.7, 0.8));
        // albedo = (h < 0.7 ? vec3(0.8, 0.7, 0.6) : (h < 0.8 ? vec3(0.8, 0.2, 0.2) : (h < 0.9 ? vec3(0.8, 0.6, 0.3) : vec3(0.5, 0.7, 0.8))));
        albedo = cols[int(h * 3.0)];
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
    // float visible = 1.0;
    // primary shading
    vec3 shaded = vec3(0);
    // directional light
    shaded += visible * Microfacet_BRDF(albedo, metallic, roughness, DirectionalLight, -rd, N) * k12000 * 0.5;
    // point light
    vec3 L = (isfloorB ? vec3(-1.4, -0.9, 2.9) : vec3(sign(hash.x - 0.5), -0.06, 0)) - RP;
    float l = length(L);
    vec3 lcol = plcol * pow(1.0 / (1.0 + max(0.0, l - 1.0)), 2.0);
    shaded += Microfacet_BRDF(albedo, metallic, roughness, L / l, -rd, N) * lcol;
    // ao
    shaded *= sqrt(saturate(sdf(P + N * 0.05) / 0.05));
    // unlit
    shaded = mix(shaded, albedo, type);
    // sky
    vec3 sky = vec3(mix(0.01, 0.3, saturate((rd.y + 0.5) * 0.5))) * smoothstep(1.0, 0.9, rd.y);
    vec3 col = mix(sky, shaded, hit.x);

    // 
    // .%%%%%....%%%%....%%%%...%%%%%%..%%%%%...%%%%%....%%%%....%%%%...%%%%%%...%%%%....%%%%..
    // .%%..%%..%%..%%..%%........%%....%%..%%..%%..%%..%%..%%..%%..%%..%%......%%......%%.....
    // .%%%%%...%%..%%...%%%%.....%%....%%%%%...%%%%%...%%..%%..%%......%%%%.....%%%%....%%%%..
    // .%%......%%..%%......%%....%%....%%......%%..%%..%%..%%..%%..%%..%%..........%%......%%.
    // .%%.......%%%%....%%%%.....%%....%%......%%..%%...%%%%....%%%%...%%%%%%...%%%%....%%%%..
    // ........................................................................................
    // 
    // 拍 t*128/60
    // float bt = fract(Time*32/60);
    // col += min(1,dot(vec3(1),abs(fwidth(N))))*smoothstep(0.5,0.0,abs(fract(length(P-ro)-Time)));

    // noise乗せた方が雰囲気いいかも
    col += h3 * 0.03;
    // 嘘 Gamma Correction
    col = pow(col, vec3(0.8));
    // aces
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    col = (col * (a * col + b)) / (col * (c * col + d) + e);
    // カラグレ
    // col.rg = smoothstep(-.1, 1., col.rg);
    // トランジション
    col *= min((0.5 - abs(lt - 0.5)) * 10, 1);
    // TAA&MotionBlur
    const float ema = 0.7;
    vec4 now = vec4(col, 1.0);
    vec4 back = texture(backBuffer0, uv);
    outColor0 = mix(now, back, ema);
}
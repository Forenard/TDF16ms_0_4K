#version 430

// 
// .%%..%%..%%..%%..%%%%%%..%%%%%%...%%%%...%%%%%...%%...%%.
// .%%..%%..%%%.%%....%%....%%......%%..%%..%%..%%..%%%.%%%.
// .%%..%%..%%.%%%....%%....%%%%....%%..%%..%%%%%...%%.%.%%.
// .%%..%%..%%..%%....%%....%%......%%..%%..%%..%%..%%...%%.
// ..%%%%...%%..%%..%%%%%%..%%.......%%%%...%%..%%..%%...%%.
// .........................................................
// 

layout(binding = 0) uniform sampler2D backBuffer0;
layout(binding = 1) uniform sampler2D backBuffer1;
layout(binding = 2) uniform sampler2D backBuffer2;
layout(binding = 3) uniform sampler2D backBuffer3;
layout(location = 0) uniform int waveOutPosition;
#if defined(EXPORT_EXECUTABLE)
vec2 resolution = vec2(SCREEN_XRESO, SCREEN_YRESO);
#define NUM_SAMPLES_PER_SEC 48000.
float time = waveOutPosition / NUM_SAMPLES_PER_SEC;
#else
layout(location = 2) uniform float time;
layout(location = 3) uniform vec2 resolution;
#endif

layout(location = 0) out vec4 outColor0;
layout(location = 1) out vec4 outColor1;
layout(location = 2) out vec4 outColor2;
layout(location = 3) out vec4 outColor3;

// 
// ..%%%%...%%.......%%%%...%%%%%....%%%%...%%.....
// .%%......%%......%%..%%..%%..%%..%%..%%..%%.....
// .%%.%%%..%%......%%..%%..%%%%%...%%%%%%..%%.....
// .%%..%%..%%......%%..%%..%%..%%..%%..%%..%%.....
// ..%%%%...%%%%%%...%%%%...%%%%%...%%..%%..%%%%%%.
// ................................................
// 
#define LoopMax 256
#define LenMax 1000.0
#define NormalEPS 1e-4
#define DistMin 1e-3

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

#define remap(x,a,b,c,d) ((((x)-(a))/((b)-(a)))*((d)-(c))+(c))
#define remapc(x,a,b,c,d) clamp(remap(x,a,b,c,d),min(c,d),max(c,d))
#define saturate(x) clamp(x,0.0,1.0)
#define linearstep(a, b, x) min(max(((x) - (a)) / ((b) - (a)), 0.0), 1.0)
#define rep(i,n) for(int i=0;i<n;i++)

mat2 rot(float a)
{
    float c = cos(a);
    float s = sin(a);
    return mat2(c, -s, s, c);
}

vec2 orbit(float a)
{
    return vec2(cos(a), sin(a));
}

vec2 fold(vec2 p, float a)
{
    vec2 v = orbit(a);
    p -= 2.0 * min(0.0, dot(p, v)) * v;
    return p;
}

vec2 pmod(vec2 suv, float div)
{
    const float shift = 0.0;
    float a = atan(suv.y, suv.x) + TAU - PI * 0.5 + PI / div;
    a = mod(a, TAU / div) + PI * 0.5 - PI / div - shift;
    return orbit(a) * length(suv);
}

// triplanar mapping
// c is world space noise (or etc)
// c.x=noise(P.zy), c.y=noise(P.xz), c.z=noise(P.xy)
float trip(vec3 c, vec3 N)
{
    const float power = 1.0;
    N = pow(abs(N), vec3(power));
    return dot(c, N / dot(vec3(1), N));
}

// hacky version
vec2 uvtrip(vec3 P, vec3 N)
{
    const float power = 1.0;
    N = sign(N) * pow(abs(N), vec3(power));
    N = N / dot(vec3(1), N);
    return N.x * P.zy + N.y * P.xz + N.z * P.xy;
}

// ortho basis
// https://en.wikipedia.org/wiki/Osculating_plane
mat3 getBNT(vec3 T)
{
    // camera rotation (may not be needed)
    // float cr = 0.0;
    // vec3 N = vec3(sin(cr), cos(cr), 0.0);
    T = normalize(T);
    vec3 N = vec3(0, 1, 0);
    vec3 B = normalize(cross(N, T));
    N = normalize(cross(T, B));
    return mat3(B, N, T);
}

// 
// .%%..%%...%%%%...%%%%%%...%%%%...%%%%%%.
// .%%%.%%..%%..%%....%%....%%......%%.....
// .%%.%%%..%%..%%....%%.....%%%%...%%%%...
// .%%..%%..%%..%%....%%........%%..%%.....
// .%%..%%...%%%%...%%%%%%...%%%%...%%%%%%.
// ........................................
// 

// http://www.jcgt.org/published/0009/03/02/
uvec3 pcg3d(uvec3 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> 16u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return v;
}
vec3 pcg33(vec3 v)
{
    uvec3 u = pcg3d(floatBitsToUint(v));
    return vec3(u) / float(0xffffffffu);
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
    rep(i, N)
    {
        v += a * vec4(perlin32(p), 1);
        a *= 0.5;
        p *= R;
    }
    return v.xyz / v.w;
}
// Voronoi - distances by iq
// https://www.shadertoy.com/view/ldl3W8
float vnoise12(vec2 x)
{
    const float magic = 1.98;
    vec2 n = floor(x);
    vec2 f = fract(x);

    vec2 mg, mr;

    float md = 8.0;
    rep(i, 9)
    {
        vec2 g = vec2(i % 3 - 1, i / 3 - 1);
        vec2 o = pcg33(vec3(n + g, magic)).xy;
        vec2 r = g + o - f;
        float d = dot(r, r);

        if(d < md)
        {
            md = d;
            mr = r;
            mg = g;
        }
    }

    md = 8.0;
    rep(i, 25)
    {
        vec2 g = vec2(i % 5 - 2, i / 5 - 2);
        vec2 o = pcg33(vec3(n + g, magic)).xy;
        vec2 r = g + o - f;

        if(dot(mr - r, mr - r) > 0.00001)
            md = min(md, dot(0.5 * (mr + r), normalize(r - mr)));
    }

    // return vec3(md, mr);
    return md;
}
// fractal of voronoi(p+fbm)
float cracknoise12(vec2 p, float g)
{
    const int N = 3;
    const mat2 R = rot(GOLD) * 1.5;

    float a = 1.0;
    vec2 v = vec2(0);
    rep(i, N)
    {
        // scalling
        vec2 q = p + fbm32(p * 1.5).xy;
        float n = vnoise12(q);
        n = smoothstep(g, 0.0, n);
        v += a * vec2(n, 1);
        a *= 0.5;
        p *= R;
    }
    return 1.0 - v.x / v.y;
}
// Cyclic Noise by nimitz (explained by jeyko)
// https://www.shadertoy.com/view/3tcyD7
// And edited by 0b5vr
// https://scrapbox.io/0b5vr/Cyclic_Noise
vec3 cyclic(vec3 p, float freq)
{
    const mat3 bnt = getBNT(vec3(1, 2, 3));
    vec4 n = vec4(0);

    for(int i = 0; i < 8; i++)
    {
        p += sin(p.yzx);
        n += vec4(cross(cos(p), sin(p.zxy)), 1);
        p *= bnt * freq;
    }
    return n.xyz / n.w;
}

// 
// .%%%%%...%%%%%...%%%%%...%%%%%%.
// .%%..%%..%%..%%..%%..%%..%%.....
// .%%%%%...%%%%%...%%..%%..%%%%...
// .%%..%%..%%..%%..%%..%%..%%.....
// .%%%%%...%%..%%..%%%%%...%%.....
// ................................
// 

// type: 0:pbr, 1:unlit(emissive)
const int MAT_PBR = 0;
const int MAT_UNLIT = 1;
struct Material
{
    int type;
    vec3 albedo;
    float metallic;
    float roughness;
};
#define Material() Material(0,vec3(1), 0.5, 0.5)

/*
ref : https://google.github.io/filament/Filament.html#materialsystem
Specular Microfacet BRDF for Realtime Rendering
*/

float pow5(float x)
{
    return (x * x) * (x * x) * x;
}

/*
Normal distribution function
(Trowbridge-Reitz distribution)
*/
float D_GGX(float roughness, float NoH)
{
    float a = NoH * roughness;
    float k = roughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1.0 / PI);
}
/*
Visibility function
(height-correlated Smith function)
*/
float V_Smith(float roughness, float NoV, float NoL)
{
    float a2 = roughness * roughness;
    float G_V = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float G_L = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (G_V + G_L);
}
float V_Smith_Fast(float roughness, float NoV, float NoL)
{
    float a = roughness;
    float G_V = NoV / (NoV * (1.0 - a) + a);
    float G_L = NoL / (NoL * (1.0 - a) + a);
    return 0.5 / (G_V + G_L);
}
/*
Fresnel function
(Schlick approximation)
F : Fresnel reflectance
F90 = 1.0
*/
vec3 F_Schlick(vec3 f0, float c)
{
    float k = pow5(1.0 - c);
    return f0 + (1.0 - f0) * k;
}
float F_Schlick_Burley(float f90, float c)
{
    const float f0 = 1.0;
    float k = pow5(1.0 - c);
    return f0 + (f90 - f0) * k;
}
/*
Disney diffuse BRDF
*/
float Fd_Burley(float roughness, float NoV, float NoL, float LoH)
{
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float ls = F_Schlick_Burley(NoL, f90);
    float vs = F_Schlick_Burley(NoV, f90);
    return ls * vs * (1.0 / PI);
}
/*
Cook-Torrance approximation
Specular-BRDF=D*G*F/(4*dot(L,N)*dot(V,N))
=D*V*F

D: Normal distribution function
G: geometric shadowing function
F: Fresnel function
V: Visibility function
*/
vec3 Microfacet_BRDF(Material mat, vec3 L, vec3 V, vec3 N, bool isSecondary)
{
    // i think 0.5
    const float reflectance = 0.5;
    vec3 albedo = mat.albedo;
    float metallic = mat.metallic;
    float paramRoughness = mat.roughness;

    float roughness = paramRoughness * paramRoughness;
    // clamp roughness
    roughness = max(roughness, 1e-3);
    vec3 f0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + albedo * metallic;

    vec3 H = normalize(L + V);
    float NoV = abs(dot(N, V)) + 1e-5;
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));
    float LoH = saturate(dot(L, H));

    // Calc specular
    float D_spec = D_GGX(roughness, NoH);
    float V_spec = V_Smith(roughness, NoV, NoL);
    // float V_spec = V_Smith_Fast(roughness, NoV, NoL);
    vec3 F_spec = F_Schlick(f0, LoH);
    vec3 Fr = (isSecondary ? 1.0 : D_spec * V_spec) * F_spec;
    // Calc diffuse
    vec3 diffColor = albedo * (1.0 - metallic);
    vec3 Fd = diffColor * Fd_Burley(roughness, NoV, NoL, LoH);

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

float sdBox(vec3 p, vec3 b)
{
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

float sdPlane(vec3 p, vec3 n, float h)
{
    return dot(p, n) + h;
}

float sdWall(vec3 p)
{
    const vec3 RoomSize = vec3(2, 1, 4);
    p.xy = mod(p.xy, RoomSize.xy) - 0.5 * RoomSize.xy;
    // wall
    float d = -p.z;
    d = max(d, -sdBox(p, RoomSize * 0.4));
    return d;
}

void opSDFMin(float sdf, inout float d, inout int mid)
{
    if(sdf < d)
    {
        d = sdf;
        MatID = mid;
    }
    mid++;
}

float sdf(vec3 p)
{
    int mid = 0;
    MatID = -1;
    float d = LenMax;

    opSDFMin(sdWall(p), d, mid);

    return d;
}

vec3 getNormal(vec3 p)
{
    const float h = NormalEPS;
    const vec2 k = vec2(1, -1);
    // keep matid
    int mid = MatID;
    vec3 n = normalize(k.xyy * sdf(p + k.xyy * h) + k.yyx * sdf(p + k.yyx * h) + k.yxy * sdf(p + k.yxy * h) + k.xxx * sdf(p + k.xxx * h));
    MatID = mid;
    return n;
}

vec3 boxDist(vec3 rp, vec3 rd, vec3 size)
{
    // shifted center
    vec3 irp = (floor((rp + sign(rd) * size) / size) + 0.5) * size;
    vec3 d = abs(irp - rp) - 0.5 * size;
    d = max(d, 0.0);
    d /= abs(rd);
    return d;
}

bool march(vec3 rd, vec3 ro, out vec3 rp)
{
    float dist, len = 0.0;
    rep(i, LoopMax)
    {
        rp = ro + rd * len;
        dist = sdf(rp);

        vec3 bd = boxDist(rp, rd, vec3(2));
        dist = min(dist, bd.z + DistMin);

        len += dist;
        if(len > LenMax)
        {
            break;
        }
        if(dist < DistMin)
        {
            return true;
        }
    }
    return false;
}

// 
// .%%...%%...%%%%...%%%%%%..%%%%%%..%%%%%...%%%%%%...%%%%...%%.....
// .%%%.%%%..%%..%%....%%....%%......%%..%%....%%....%%..%%..%%.....
// .%%.%.%%..%%%%%%....%%....%%%%....%%%%%.....%%....%%%%%%..%%.....
// .%%...%%..%%..%%....%%....%%......%%..%%....%%....%%..%%..%%.....
// .%%...%%..%%..%%....%%....%%%%%%..%%..%%..%%%%%%..%%..%%..%%%%%%.
// .................................................................
// 

Material matConcrete(vec3 P, inout vec3 N)
{
    Material mat = Material();
    vec2 uv = uvtrip(P, N);

    vec3 fbm = fbm32(uv * 3.0 * vec2(3, 1));// gravity ydown
    vec3 fbm2 = fbm32(uv * 96.0);
    float crn = cracknoise12(uv * 2.0, 0.02);
    // base*detail*crack
    float bc = mix(0.6, 1.0, pow(fbm.y, 1.0)) * mix(0.8, 1.0, pow(fbm2.x, 3.0)) * pow(crn, 0.5);
    // waku
    vec2 auv = abs(fract(uv + (fbm2.yz - 0.5) * 0.01) - 0.5);
    const float wakw = 0.005;
    float wak = (1.0 - smoothstep(0.5 - wakw * 0.5, 0.5 - wakw, auv.x) * smoothstep(0.5 - wakw * 0.5, 0.5 - wakw, auv.y)) * fbm2.y;
    wak = mix(1.0, 0.2, wak);
    bc *= wak;
    // scrach
    vec3 cyc = cyclic(P * 3.0, 1.5);
    float scr = smoothstep(0.3, 0.7, cyc.z);
    bc *= mix(1.0, 0.7, scr);
    // color
    // const vec3 bcol = vec3(1), scol = vec3(108, 100, 89) / 150.0;
    // mat.albedo = mix(bcol, scol, pow(fbm.z, 3.0)) * bc;
    mat.albedo = saturate(vec3(1.3) * bc);
    mat.roughness = mix(0.5, 1.0, pow(fbm.y, 3.0));
    mat.metallic = 0.1;
    // normal map
    N = normalize(N + (fbm * 2.0 - 1.0) * 0.03 + cyc * 0.02);
    return mat;
}

Material getMaterial(vec3 P, inout vec3 N)
{
    Material mat = Material();

    if(MatID == 0)
    {
        mat = matConcrete(P, N);
    }

    return mat;
}

// 
// ..%%%%...%%..%%...%%%%...%%%%%...%%%%%%..%%..%%...%%%%..
// .%%......%%..%%..%%..%%..%%..%%....%%....%%%.%%..%%.....
// ..%%%%...%%%%%%..%%%%%%..%%..%%....%%....%%.%%%..%%.%%%.
// .....%%..%%..%%..%%..%%..%%..%%....%%....%%..%%..%%..%%.
// ..%%%%...%%..%%..%%..%%..%%%%%...%%%%%%..%%..%%...%%%%..
// ........................................................
// 

// 1/(r^2+1) * saturate((1-(r/rmax)^2)^2)
void pointLighting(vec3 P, vec3 lpos, float lmin, float lmax, out vec3 L, out float lint)
{
    L = lpos - P;
    float r = length(L);
    L /= r;
    r = max(0., r - lmin);
    float c = 1.0 / (r * r + 1.0);
    float w = r / lmax;
    w = 1.0 - w * w;
    w = saturate(w * w);
    lint = c * w;
}

vec3 directionalLighting(Material mat, vec3 P, vec3 V, vec3 N)
{
    vec3 L = normalize(vec3(-1, 1, -1));
    vec3 lcol = vec3(1);
    return Microfacet_BRDF(mat, L, V, N, false) * lcol;
}

vec3 lighting(Material mat, vec3 P, vec3 V, vec3 N)
{
    vec3 col = vec3(0);
    col += directionalLighting(mat, P, V, N);
    return col;
}

vec3 secondaryShading(vec3 P, vec3 V, vec3 N)
{
    Material mat = getMaterial(P, N);
    return mat.albedo * float(mat.type == MAT_UNLIT);

    // if(mat.type == MAT_UNLIT)
    // {
    //     return mat.albedo;
    // }
    // return lighting(mat, P, V, N);
}

vec3 shading(inout vec3 P, vec3 V, vec3 N)
{
    Material mat = getMaterial(P, N);
    if(mat.type == MAT_UNLIT)
    {
        // unlit
        return mat.albedo;
    }

    // avoid self-intersection
    P += N * DistMin;
    // primary shading
    vec3 col = lighting(mat, P, V, N);

    // secondary ray
    vec3 SP;
    vec3 srd = reflect(-V, N);
    vec3 sro = P;
    if(!march(srd, sro, SP))
    {
        return col;
    }
    vec3 SV = -srd;
    vec3 SN = getNormal(SP);
    // avoid self-intersection
    SP += SN * DistMin;
    // secondary shading
    vec3 scol = secondaryShading(SP, SV, SN);

    // fake reflection
    // roughness補正が必要かも
    col += scol * Microfacet_BRDF(mat, srd, V, N, true);
    // bad
    // col += scol * Microfacet_BRDF(mat, srd, V, N, true)*mat.metallic;

    return col;
}

// 
// .%%%%%....%%%%....%%%%...%%%%%%.
// .%%..%%..%%..%%..%%..%%....%%...
// .%%%%%...%%..%%..%%..%%....%%...
// .%%..%%..%%..%%..%%..%%....%%...
// .%%..%%...%%%%....%%%%.....%%...
// ................................
// 

// primary ray
vec3 tracer(vec3 rd, vec3 ro)
{
    vec3 rp;
    if(!march(rd, ro, rp))
    {
        return vec3(0);// TODO:sky
    }
    vec3 N = getNormal(rp);
    vec3 col = shading(rp, -rd, N);

    return col;
}

void getUV(out vec2 uv, out vec2 suv)
{
    vec2 fc = gl_FragCoord.xy, res = resolution.xy, asp = res / min(res.x, res.y);
    uv = fc / res;
    suv = (uv * 2.0 - 1.0) * asp;
    // Set Time
    Time = time;
}

void getRORD(out vec3 ro, out vec3 rd, out vec3 dir, vec2 suv)
{
    // Parameter
    float fov = 60.0;
    float fisheye = 0.0;
    // 0.0: ortho, 1.0: persp
    float persp = 1.0;
    // ortho size
    float osize = 2.5;
    // ortho near
    float onear = -5.0;

    float lt = mod(Time, 6.0);
    if(lt < 3.0)
    {
        ro = vec3(0, Time, -5);
        dir = normalize(vec3(-0.5, 0.5, 1));
    }
    else
    {
        ro = vec3(0, Time * 0.2, -1);
        dir = normalize(vec3(-0.5, 0.5, 1));
    }

    mat3 bnt = getBNT(dir);
    float zf = 1.0 / tan(fov * PI / 360.0);
    zf -= zf * length(suv) * fisheye;

    vec3 prd, pro, ord, oro;
    {
        // perspective
        prd = normalize(bnt * vec3(suv, zf));
        pro = ro;
    }
    {
        // ortho
        ord = dir;
        oro += bnt * vec3(suv * osize, onear);
    }

    rd = normalize(mix(ord, prd, persp));
    ro = mix(oro, pro, persp);
}

vec3 overlay(vec3 bcol, vec2 uv, vec2 suv)
{
    return bcol;
}

vec3 postProcess(vec3 bcol, vec2 uv, vec2 suv)
{
    return bcol;
}

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
    // Get UV
    vec2 uv, suv;
    getUV(uv, suv);

    // Camera
    vec3 ro, rd, dir;
    getRORD(ro, rd, dir, suv);

    // Tracer
    vec3 col = tracer(rd, ro);

    // Overlay
    col = overlay(col, uv, suv);

    // Post Process
    col = postProcess(col, uv, suv);

    outColor0 = vec4(col, 1);
}
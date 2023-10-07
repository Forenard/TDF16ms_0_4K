#version 430

#define saturate(i) clamp(i,0.,1.)
#define linearstep(a,b,x) saturate(((x)-(a))/((b)-(a)))
#define p2f(i) (exp2(((i)-69.)/12.)*440.)
#define rep(i,n) for(int i=0;i<(n);i++)
#define inRange(t,a,b) (step(a,t)*(1.-step(b,t)))
#define inRangeB(t,a,b) ((a<=t)&&(t<b))

const float PI = acos( -1.0 );
const float TAU = PI * 2.0;
const float GOLD = PI * (3.0 - sqrt(5.0));// 2.39996...

const float BPS = 2.1;
const float B2T = 1.0 / BPS;
const float S2T = 0.25 * B2T;
const float SAMPLES_PER_SEC = 48000.0;

int SAMPLES_PER_STEP = int( SAMPLES_PER_SEC / BPS / 4.0 );
int SAMPLES_PER_BEAT = 4 * SAMPLES_PER_STEP;

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
    return vec3(u) / float(-1u);
}

vec2 orbit( float t ) {
  return vec2( cos( t ), sin( t ) );
}

mat2 rot( float x ) {
  vec2 v = orbit( x );
  return mat2( v.x, v.y, -v.y, v.x );
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

vec2 boxMuller( vec2 xi ) {
  float r = sqrt( -2.0 * log( xi.x ) );
  float t = xi.y;
  return r * orbit( TAU * t );
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

vec3 cyclic( vec3 p, float pump ) {
  vec4 sum = vec4( 0 );
  mat3 rot = getBNT( vec3( 2, -3, 1 ) );

  rep( i, 5 ) {
    p *= rot;
    p += sin( p.zxy );
    sum += vec4( cross( cos( p ), sin( p.yzx ) ), 1 );
    sum *= pump;
    p *= 2.0;
  }

  return sum.xyz / sum.w;
}

layout(location = 0) uniform int waveOutPosition;

#if defined(EXPORT_EXECUTABLE)
  /*
    shader minifier が compute シェーダに対応していない問題を回避するためのハック。
    以下の記述はシェーダコードとしては正しくないが、shader minifier に認識され
    minify が適用されたのち、work_around_begin: 以降のコードに置換される。
    %s は、shader minifier によるリネームが適用されたあとのシンボル名に
    置き換えらえる。
  */
  #pragma work_around_begin:layout(std430,binding=0)buffer _{vec2 %s[];};layout(local_size_x=1)in;
  vec2 waveOutSamples[];
  #pragma work_around_end
#else
  layout(std430, binding = 0) buffer SoundOutput{ vec2 waveOutSamples[]; };
  layout(local_size_x = 1) in;
#endif

vec2 shotgun( float t, float spread ) {
  vec2 sum = vec2( 0.0 );

  rep( i, 64 ) {
    vec3 dice = pcg33( vec3( i ) );
    sum += vec2( sin( TAU * t * exp2( spread * dice.x ) ) ) * rot( TAU * dice.y );
  }

  return sum / 64.0;
}

vec4 seq16( int seq, float st ) {
  int sti = int( st );
  int rotated = ( ( seq >> ( 15 - sti ) ) | ( seq << ( sti + 1 ) ) ) & 0xffff;

  float prevStepBehind = log2( float( rotated & -rotated ) );
  float prevStep = float( sti ) - prevStepBehind;
  float nextStepForward = 16.0 - floor( log2( float( rotated ) ) );
  float nextStep = float( sti ) + nextStepForward;

  return vec4(
    prevStep,
    st - prevStep,
    nextStep,
    nextStep - st
  );
}

float cheapFilterSaw( float phase, float k ) {
  float wave = fract( phase );
  float c = smoothstep( 1.0, 0.0, wave / k );
  return ( wave + c ) * 2.0 - 1.0 - k;
}

float CHORDS[] = float[](
  0.0, 2.0, 3.0, 7.0, 14.0, 15.0, 19.0, 22.0,
  -5.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 15.0,
  -4.0, 0.0, 3.0, 7.0, 14.0, 15.0, 19.0, 24.0,
  -7.0, 0.0, 7.0, 8.0, 10.0, 12.0, 15.0, 19.0
);

void main() {
  int frame = int( gl_GlobalInvocationID.x ) + waveOutPosition;
  vec4 time = vec4( ( frame ) % ( SAMPLES_PER_BEAT * ivec4( 1, 4, 64, 65536 ) ) ) / SAMPLES_PER_SEC;
  vec4 beats = time * BPS;

  // return float( max( 0, frame + SAMPLES_PER_STEP * offset ) % ( SAMPLES_PER_STEP * every ) ) / SAMPLES_PER_SEC;

  int prog = beats.w >= 128.0
    ? ( int( time.w / 8.0 * BPS ) * 8 % 32 )
    : 0;

  vec2 dest = vec2( 0.0 );
  float sidechain = 0.0;

  { // kick
    float t = float( frame % SAMPLES_PER_BEAT ) / SAMPLES_PER_SEC;
    float q = B2T - t;
    sidechain = smoothstep( 0.0, 0.4, t ) * smoothstep( 0.0, 0.001, q );

    float env = smoothstep( 0.0, 0.001, q ) * smoothstep( 0.3, 0.1, t );

    float wave = sin(
      270.0 * t
      - 20.0 * exp( -20.0 * t )
      - 15.0 * exp( -80.0 * t )
      - 10.0 * exp( -500.0 * t )
    );
    dest += 0.6 * tanh( 2.0 * env * wave );
  }

  { // bass
    float t = float( frame % SAMPLES_PER_STEP ) / SAMPLES_PER_SEC;
    float q = S2T - t;

    float env = smoothstep( 0.0, 0.001, t ) * smoothstep( 0.0, 0.001, q );

    float k = 0.55 - 0.4 * exp( -5.0 * t );

    {
      float phase = t * p2f( 30.0 + CHORDS[ prog ] );
      dest += 0.4 * sidechain * env * tanh( 2.0 * sin( TAU * phase ) );
    }

    rep( i, 8 ) {
      vec3 dice = pcg33( vec3( i + 51 ) );
      vec2 dicei = boxMuller( dice.xy );
      float phase = t * p2f( 30.0 + CHORDS[ prog ] + 0.2 * dicei.x ) + dice.z;

      float wave = tanh( 3.0 * cheapFilterSaw( phase, k ) );
      dest += 0.1 * sidechain * tanh( 2.0 * env * wave );
    }
  }

  if ( inRangeB( beats.w, 32.0, 128.0 ) || beats.w >= 192.0 ) { // hihat
    float st = floor( frame / SAMPLES_PER_STEP % 16 );
    float t = float( frame % SAMPLES_PER_STEP ) / SAMPLES_PER_SEC;
    float q = S2T - t;

    float decay = exp2( 5.0 + fract( 0.8 + 0.631 * vec3( st ) ).x );
    float env = smoothstep( 0.0, 0.001, q ) * exp( -decay * t );

    vec2 wave = shotgun( 6000.0 * t, 2.0 );
    dest += 0.3 * mix( 0.4, 1.0, sidechain ) * tanh( 2.0 * env * wave );
  }

  if ( inRangeB( beats.w, 64.0, 128.0 ) || beats.w >= 256.0 ) { // clap
    float t = mod( time.y - B2T, 2.0 * B2T );

    float env = mix(
      exp( -80.0 * mod( t, 0.02 ) ),
      exp( -20.0 * t ),
      smoothstep( 0.02, 0.04, t )
    );

    vec2 wave = cyclic( vec3( 4.0 * orbit( 1000.0 * t ), 480.0 * t ), 2.0 ).xy;

    dest += 0.1 * tanh( 3.0 * env * wave );
  }

  if ( inRangeB( beats.w, 64.0, 128.0 ) || beats.w >= 192.0 ) { // perc
    rep( i, 16 ) {
      vec3 dice = pcg33( vec3( i ) + 4.995 );
      vec3 dice2 = pcg33( dice );

      float t = time.z;
      t = mod(
        mod( t, 8.0 * B2T ) - 0.25 * B2T * float( i ),
        0.25 * B2T * floor( 3.0 + 14.0 * dice.x )
      );

      float env = exp( -exp2( 5.0 + 3.0 * dice.y ) * t );

      float freq = exp2( 8.0 + 8.0 * dice2.x );
      vec2 wave = cyclic( vec3(
        exp2( -2.0 + 4.0 * dice2.y ) * orbit( freq * t ),
        exp2( -2.0 + 5.0 * dice2.z ) * freq * t
      ), 2.0 ).xy;

      dest += 0.08 * sidechain * tanh( 3.0 * env * wave );
    }
  }

  rep( i, 64 ) { // chords
    vec3 dice = pcg33( vec3( i ) );
    vec2 dicen = boxMuller( dice.xy );

    float pog = smoothstep( 160.0, 224.0, beats.w );
    int foff = int( exp2( 9.2 * pog ) );
    float t = float( ( frame - foff * i ) % ( 3 * SAMPLES_PER_STEP ) ) / SAMPLES_PER_SEC;
    float q = 0.75 * B2T - t;
    float env = smoothstep( 0.0, 0.001, t ) * smoothstep( 0.0, 0.001, q ) * exp( -3.0 * t );

    int chordn = int( mix( 4.0, 8.0, linearstep( 128.0, 192.0, beats.w ) ) );
    float freq = p2f( 54.0 + CHORDS[ i % chordn + prog ] + 0.1 * clamp( dicen.x, -1.0, 1.0 ) );
    float phase = freq * time.z + dice.x;

    { // pluck
      float k = 1.0 - mix( 0.9, 0.99, pog ) * exp( -mix( 3.0, 1.0, pog ) * t );
      vec2 wave = vec2( cheapFilterSaw( phase, k ) );
      dest += 0.03 * mix( 0.2, 1.0, sidechain ) * env * wave * rot( float( i ) );
    }
  }

  if ( beats.w >= 128.0 ) { // arp
    vec2 sum=vec2(0);

    rep( i, 64 ) {
      float t = float( ( frame - i * SAMPLES_PER_STEP ) % ( 64 * SAMPLES_PER_STEP ) ) / SAMPLES_PER_SEC;

      float fi = float( i );
      float t2f = mod( t, 0.75 * B2T );
      float t2i = floor( t / ( 0.75 * B2T ) );
      float q = 0.75 * B2T - t2f;
      float env = smoothstep( 0.0, 0.002, t2f ) * smoothstep( 0.0, 0.01, q ) * exp(-12.*t2f) * exp(-t2i);

      int arpseed = int( 21.0 * fract( 0.825 * fi ) );
      float note = 42.0 + CHORDS[ arpseed % 8 + prog ] + 12.0 * float( arpseed / 8 );
      float freq = p2f(note);
      vec2 phase = t * freq + pcg33( vec3( fi ) ).xy;
      phase += 0.4 * fbm32( vec2( 0.1 * phase.x ) ).xy;

      vec2 wave = vec2( sin(
        TAU * phase
      ) );

      sum+=2.*env*wave*rot(time.w);
    }

    dest+=.1*mix(.2,1.,sidechain)*sum;
  }

  waveOutSamples[ frame ] = dest;
}


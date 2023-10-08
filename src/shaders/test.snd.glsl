#version 430

const float PI = acos( -1.0 );
const float TAU = PI * 2.0;
const float GOLD = PI * (3.0 - sqrt(5.0));// 2.39996...

const float SAMPLES_PER_SEC = 48000.0;

const float STEP2TIME = 60.0 / 128.0 / 4.0;
const int SAMPLES_PER_STEP = 5625;

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
    for (int i = 0; i < N; i ++)
    {
        v += a * vec4(perlin32(p), 1);
        a *= 0.5;
        p *= R;
    }
    return v.xyz / v.w;
}

vec2 i_boxMuller( vec2 xi ) {
  float i_r = sqrt( -2.0 * log( xi.x ) );
  float i_t = xi.y;
  return i_r * orbit( TAU * i_t );
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

float i_p2f( float p ) {
  return exp2( ( p - 69.0 ) / 12.0 ) * 440.0;
}

float i_cheapFilterSaw( float phase, float k ) {
  float i_wave = fract( phase );
  float i_c = smoothstep( 1.0, 0.0, i_wave / k );
  return ( i_wave + i_c ) * 2.0 - 1.0 - k;
}

float CHORDS[] = float[](
  0.0, 2.0, 3.0, 7.0, 14.0, 15.0, 19.0, 22.0,
  -5.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 15.0,
  -4.0, 0.0, 3.0, 7.0, 14.0, 15.0, 19.0, 24.0,
  -7.0, 0.0, 7.0, 8.0, 10.0, 12.0, 15.0, 19.0
);

void main() {
  int frame = int( gl_GlobalInvocationID.x ) + waveOutPosition;
  vec4 time = vec4( ( frame ) % ( SAMPLES_PER_STEP * ivec4( 4, 16, 256, 65536 ) ) ) / SAMPLES_PER_SEC;
  vec4 beats = time / 4.0 / STEP2TIME;

  // return float( max( 0, frame + SAMPLES_PER_STEP * offset ) % ( SAMPLES_PER_STEP * every ) ) / SAMPLES_PER_SEC;

  // -- tenkai -------------------------------------------------------------------------------------
  const int TENKAI_PROG_STEP = 128 * 4;
  const int TENKAI_CHORD_START_STEP = 128 * 4;
  const int TENKAI_CHORD_LENGTH_STEP = 16 * 4;
  const int TENKAI_ARP_START_STEP = 0;//64 * 4;

  bool i_tenkaiKickActive = beats.w >= 64.0;
  bool i_tenkaiBassActive = beats.w >= 32.0;
  bool i_tenkaiCrashActive = beats.w >= 32.0;
  bool i_tenkaiHihatActive = ( beats.w >= 32.0 && beats.w < 128.0 ) || beats.w >= 192.0;
  bool i_tenkaiClapActive = ( beats.w >= 64.0 && beats.w < 128.0 ) || beats.w >= 256.0;
  bool i_tenkaiPercActive = ( beats.w >= 64.0 && beats.w < 128.0 ) || beats.w >= 192.0;
  float i_pluckOffset = smoothstep( 160.0, 224.0, beats.w );
  float i_pluckFilterEnv = smoothstep( 256.0, 192.0, beats.w );
  // -- tenkai end ---------------------------------------------------------------------------------

  float t;

  int prog = max( 0, frame / SAMPLES_PER_STEP - TENKAI_PROG_STEP ) / 32 % 4 * 8;

  vec2 dest = vec2( 0.0 );
  float sidechain = 1.0;

  if ( i_tenkaiKickActive ) { // kick
    t = float( frame % ( 4 * SAMPLES_PER_STEP ) ) / SAMPLES_PER_SEC;
    float i_q = 4.0 * STEP2TIME - t;
    sidechain = 0.2 + 0.8 * smoothstep( 0.0, 0.4, t ) * smoothstep( 0.0, 0.001, i_q );

    float i_env = smoothstep( 0.0, 0.001, i_q ) * smoothstep( 0.3, 0.1, t );

    float i_wave = sin(
      270.0 * t
      - 20.0 * exp( -t * 20.0 )
      - 15.0 * exp( -t * 80.0 )
      - 10.0 * exp( -t * 500.0 )
    );
    dest += 0.6 * tanh( 2.0 * i_env * i_wave );
  }

  if ( i_tenkaiBassActive ) { // bass
    t = float( frame % SAMPLES_PER_STEP ) / SAMPLES_PER_SEC;
    float i_q = STEP2TIME - t;

    float env = smoothstep( 0.0, 0.001, t ) * smoothstep( 0.0, 0.001, i_q );

    {
      float phase = t * i_p2f( 30.0 + CHORDS[ prog ] );
      dest += 0.4 * sidechain * env * tanh( 2.0 * sin( TAU * phase ) );
    }

    for ( int i = 0; i < 8; i ++ ) {
      vec3 dice = pcg33( vec3( i + 51 ) );
      vec2 i_dicei = i_boxMuller( dice.xy );
      float phase = t * i_p2f( 30.0 + CHORDS[ prog ] + 0.2 * i_dicei.x ) + dice.z;

      float i_k = 0.55 - 0.4 * exp( -t * 5.0 );
      float i_wave = tanh( 3.0 * i_cheapFilterSaw( phase, i_k ) );
      dest += 0.1 * sidechain * tanh( 2.0 * env * i_wave );
    }
  }

  if ( i_tenkaiCrashActive ) { // crash
    t = float( frame % ( 256 * SAMPLES_PER_STEP ) ) / SAMPLES_PER_SEC;

    float env = exp( -t * 10.0 ) + exp( -t ); // don't inline me!

    // shotgun
    t *= 28000.0;
    vec2 wave = vec2( 0.0 );
    for ( int i = 0; i < 64; i ++ ) {
      vec3 dice = pcg33( vec3( i ) );
      wave += vec2( sin( t * exp2( 2.0 * dice.x ) ) ) * rot( TAU * dice.y ) / 64.0;
    }
    dest += 0.2 * sidechain * tanh( 2.0 * env * wave );
  }

  if ( i_tenkaiHihatActive ) { // hihat
    t = float( frame % SAMPLES_PER_STEP ) / SAMPLES_PER_SEC;
    float i_q = STEP2TIME - t;
    float i_st = floor( frame / SAMPLES_PER_STEP % 16 );

    float i_decay = exp2( 5.0 + fract( 0.8 + 0.631 * vec3( i_st ) ).x );
    float env = smoothstep( 0.0, 0.001, i_q ) * exp( -t * i_decay ); // don't inline me!

    // shotgun
    t *= 38000.0;
    vec2 wave = vec2( 0.0 );
    for ( int i = 0; i < 64; i ++ ) {
      vec3 dice = pcg33( vec3( i ) );
      wave += vec2( sin( t * exp2( 2.0 * dice.x ) ) ) * rot( TAU * dice.y ) / 64.0;
    }
    dest += 0.3 * sidechain * tanh( 2.0 * env * wave );
  }

  if ( i_tenkaiClapActive ) { // clap
    t = float( ( frame + 4 * SAMPLES_PER_STEP ) % ( 8 * SAMPLES_PER_STEP ) ) / SAMPLES_PER_SEC;

    float i_env = exp( -t * 16.0 );

    vec2 i_wave = fbm32( vec2( 4.0 * orbit( 1000.0 * t ) + 480.0 * t ) ).xy - 0.5;

    dest += 0.1 * tanh( 6.0 * i_env * i_wave );
  }

  if ( i_tenkaiPercActive ) { // perc
    for ( int i = 0; i < 32; i ++ ) {
      vec3 dice = pcg33( vec3( i ) + 4.995 );
      vec3 dice2 = pcg33( dice );

      int i_every = 3 + int( 14.0 * dice.x );
      t = float( ( frame % ( 32 * SAMPLES_PER_STEP ) + i * SAMPLES_PER_STEP ) % ( i_every * SAMPLES_PER_STEP ) ) / SAMPLES_PER_SEC;

      float i_env = exp( -t * exp2( 5.0 + 3.0 * dice.y ) );

      float i_freq = exp2( 8.0 + 8.0 * dice2.x );
      float phase = i_freq * t;

      vec2 i_wave = fbm32( vec2(
        exp2( -2.0 + 4.0 * dice2.y ) * orbit( phase )
        + exp2( -2.0 + 5.0 * dice2.z ) * phase
      ) ).xy - 0.5;

      dest += 0.08 * sidechain * tanh( 6.0 * i_env * i_wave );
    }
  }

  for ( int i = 0; i < 64; i ++ ) { // chords
    vec3 dice = pcg33( vec3( i ) );
    vec2 dicen = i_boxMuller( dice.xy );

    int i_frameOffset = int( exp2( 9.2 * i_pluckOffset ) ) * i + SAMPLES_PER_STEP;
    int i_frameNote = max( 0, frame - i_frameOffset ) % ( 3 * SAMPLES_PER_STEP );
    int i_frameStart = frame - i_frameNote;
    int i_prog = max( 0, i_frameStart / SAMPLES_PER_STEP - TENKAI_PROG_STEP ) / 32 % 4 * 8;

    t = float( i_frameNote ) / SAMPLES_PER_SEC;
    float i_q = 3.0 * STEP2TIME - t;
    float i_env = smoothstep( 0.0, 0.001, t ) * smoothstep( 0.0, 0.001, i_q ) * exp( -t * 3.0 );

    int chordn = 4 + clamp( ( i_frameStart / SAMPLES_PER_STEP - TENKAI_CHORD_START_STEP ) / TENKAI_CHORD_LENGTH_STEP, 0, 4 );
    float freq = i_p2f( 54.0 + CHORDS[ i % chordn + i_prog ] + 0.1 * clamp( dicen.x, -1.0, 1.0 ) );
    float phase = freq * time.z + dice.x;

    float i_k = 0.15 * i_pluckFilterEnv + 2.0 * t;
    vec2 i_wave = vec2( i_cheapFilterSaw( phase, i_k ) );
    dest += 0.03 * sidechain * i_env * i_wave * rot( float( i ) );
  }

  { // arp
    for ( int i = 0; i < 64; i ++ ) {
      t = float( max( 0, frame - ( i + TENKAI_ARP_START_STEP ) * SAMPLES_PER_STEP ) % ( 64 * SAMPLES_PER_STEP ) ) / SAMPLES_PER_SEC;

      float i_fi = float( i );
      float t2f = mod( t, 3.0 * STEP2TIME );
      float i_t2i = floor( t / ( 3.0 * STEP2TIME ) );
      float i_q = 3.0 * STEP2TIME - t2f;
      float i_env = smoothstep( 0.0, 0.002, t2f ) * smoothstep( 0.0, 0.01, i_q ) * exp( -t2f * 12.0 ) * exp( -i_t2i );

      int arpseed = int( 21.0 * fract( 0.825 * i_fi ) );
      float i_note = 42.0 + CHORDS[ arpseed % 8 + prog ] + 12.0 * float( arpseed / 8 );
      float freq = i_p2f( i_note );
      vec2 phase = (
        t * freq
        + pcg33( vec3( i_fi ) ).xy
        + 0.4 * fbm32( vec2( 0.1 * t * freq ) ).xy
      );

      vec2 i_wave = vec2( sin( TAU * phase ) );
      dest += 0.2 * sidechain * i_env * i_wave * rot( time.w );
    }
  }

  waveOutSamples[ frame ] = dest;
}

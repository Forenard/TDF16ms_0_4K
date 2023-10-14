#version 430

const float TAU = acos(-1) * 2;

const float SAMPLES_PER_SEC = 48000.;

const float STEP2TIME = 60.0 / 128 / 4;
const int SAMPLES_PER_STEP = 5625;

vec3 pcg33(vec3 v)
{
  uvec3 x = floatBitsToUint(v);
  const uint k = 1103515245u;
  x = ((x >> 8U) ^ x.yzx) * k;
  x = ((x >> 8U) ^ x.yzx) * k;
  x = ((x >> 8U) ^ x.yzx) * k;
  return vec3(x) / -1u;
}

vec2 orbit(float t)
{
  return vec2(cos(t), sin(t));
}

mat2 rot(float x)
{
  vec2 v = orbit(x);
  return mat2(v.x, v.y, -v.y, v.x);
}
vec3 perlin32(vec2 p)
{
  const float magic = 0.12;
  vec2 i = floor(p);
  vec2 f = fract(p);
    // smoothstep
  f = f * f * (3 - 2 * f);

  vec3 vx0 = mix(pcg33(vec3(i, magic)), pcg33(vec3(i + vec2(1, 0), magic)), f.x);
  vec3 vx1 = mix(pcg33(vec3(i + vec2(0, 1), magic)), pcg33(vec3(i + vec2(1, 1), magic)), f.x);
  return mix(vx0, vx1, f.y);
}
vec3 fbm32(vec2 p)
{
  const int N = 6;

  float a = 1;
  vec4 v = vec4(0);
  for(int i = 0; i < N; i++)
  {
    v += a * vec4(perlin32(p), 1);
    a *= 0.5;
        p *= mat2(-1.4747, 1.351, -1.351, -1.4747);
  }
  return v.xyz / v.w;
}

vec2 i_boxMuller(vec2 xi)
{
  float i_r = sqrt(-2 * log(xi.x));
  float i_t = xi.y;
  return i_r * orbit(TAU * i_t);
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
layout(std430, binding = 0) buffer SoundOutput
{
  vec2 waveOutSamples[];
};
layout(local_size_x = 1) in;
#endif

float i_p2f(float p)
{
  return exp2((p - 69) / 12) * 440;
}

float i_cheapFilterSaw(float phase, float k)
{
  float i_wave = fract(phase);
  float i_c = smoothstep(1.0, 0.0, i_wave / k);
  return (i_wave + i_c) * 2 - 1 - k;
}

float CHORDS[] = float[](0, 2, 3, 7, 14, 15, 19, 22, -5, 2, 3, 5, 7, 10, 14, 15, -4, 0, 3, 7, 14, 15, 19, 24, -7, 0, 7, 8, 10, 12, 15, 19);

void main()
{
  int frame = int(gl_GlobalInvocationID.x) + waveOutPosition;
  vec4 time = vec4((frame) % (SAMPLES_PER_STEP * ivec4(1, 4, 16, 256))) / SAMPLES_PER_SEC;
  float bars = frame / SAMPLES_PER_SEC / 16 / STEP2TIME;

  // -- tenkai -------------------------------------------------------------------------------------
  const int TENKAI_PROG_STEP = 16 * 16;
  const int TENKAI_CHORD_START_STEP = 16 * 16;
  const int TENKAI_CHORD_LENGTH_STEP = 4 * 16;
  const int TENKAI_ARP_START_STEP = 0;

  bool i_sidechainActive = bars >= 16;
  bool i_tenkaiKickActive = (bars >= 16 && bars < 47.25) || bars >= 48;
  bool i_tenkaiBassActive = bars >= 8;
  bool i_tenkaiCrashActive = bars >= 16;
  bool i_tenkaiCrashEvery8Bars = bars >= 64;
  bool i_tenkaiHihatActive = (bars >= 8 && bars < 16) || (bars >= 32 && bars < 72);
  bool i_tenkaiClapActive = bars >= 48 && bars < 72;
  bool i_tenkaiPercActive = bars >= 32 && bars < 72;
  float i_pluckOffset = smoothstep(34, 44, bars);
  float i_pluckFilterEnv = smoothstep(48, 32, bars);
  float i_masterAmp = smoothstep(72 + 8, 72, bars);
  // -- tenkai end ---------------------------------------------------------------------------------

  float t;

  vec2 dest = vec2(0);
  float sidechain;

  { // kick
    t = time.y;
    float i_q = 4 * STEP2TIME - t;
    sidechain = i_sidechainActive
      ? (0.2 + 0.8 * smoothstep(0.0, 0.4, t) * smoothstep(0.0, 0.001, i_q))
      : 1;

    if (i_tenkaiKickActive) {
      float i_env = smoothstep(0.0, 0.001, i_q) * smoothstep(0.3, 0.1, t);

      float i_wave = sin(270 * t - 20 * exp(-t * 20) - 15 * exp(-t * 80) - 10 * exp(-t * 500));
      dest += 0.6 * tanh(2 * i_env * i_wave);
    }
  }

  if(i_tenkaiBassActive)
  { // bass
    int prog = max(0, frame / SAMPLES_PER_STEP - TENKAI_PROG_STEP) / 32 % 4 * 8;

    t = time.x;
    float i_q = STEP2TIME - t;

    float env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, i_q);
    float i_gain = 4 * (frame % (4 * SAMPLES_PER_STEP) / SAMPLES_PER_SEC);

    {
      float phase = t * i_p2f(30 + CHORDS[prog]);
      float i_wave = sin(TAU * phase);
      dest += 0.4 * sidechain * env * tanh(i_gain * i_wave);
    }

    for(int i = 0; i < 8; i++)
    {
      vec3 dice = pcg33(vec3(i + 51));
      vec2 i_dicei = i_boxMuller(dice.xy);
      float phase = t * i_p2f(30 + CHORDS[prog] + 0.2 * i_dicei.x) + dice.z;

      float i_k = 0.55 - 0.4 * exp(-t * 5);
      float i_wave = tanh(3 * i_cheapFilterSaw(phase, i_k));
      dest += 0.1 * sidechain * tanh(i_gain * env * i_wave);
    }
  }

  if(i_tenkaiCrashActive)
  { // crash
    t = i_tenkaiCrashEvery8Bars ? mod(time.w, 8 * 16 * STEP2TIME) : time.w;

    float env = exp(-t * 10) + exp(-t); // don't inline me!

    // shotgun
    t *= 28000;
    vec2 wave = vec2(0);
    for(int i = 0; i < 64; i++)
    {
      vec3 dice = pcg33(vec3(i));
      wave += vec2(sin(t * exp2(2 * dice.x))) * rot(i) / 64;
    }
    dest += 0.2 * sidechain * tanh(2 * env * wave);
  }

  if(i_tenkaiHihatActive)
  { // hihat
    t = time.x;
    float i_q = STEP2TIME - t;
    float i_st = floor(frame / SAMPLES_PER_STEP % 16);

    float i_decay = exp2(5 + fract(0.8 + 0.631 * vec3(i_st)).x);
    float env = smoothstep(0.0, 0.001, i_q) * exp(-t * i_decay); // don't inline me!

    // shotgun
    t *= 38000;
    vec2 wave = vec2(0);
    for(int i = 0; i < 64; i++)
    {
      vec3 dice = pcg33(vec3(i));
      wave += vec2(sin(t * exp2(2 * dice.x))) * rot(i) / 64;
    }
    dest += 0.3 * sidechain * tanh(2 * env * wave);
  }

  if(i_tenkaiClapActive)
  { // clap
    t = mod(time.z + 4 * STEP2TIME, 8 * STEP2TIME);

    float i_env = exp(-t * 16);

    vec2 i_wave = fbm32(vec2(4 * orbit(1000 * t) + 480 * t)).xy - 0.5;

    dest += 0.1 * tanh(6 * i_env * i_wave);
  }

  if(i_tenkaiPercActive)
  { // perc
    for(int i = 0; i < 16; i++)
    {
      vec3 dice = pcg33(vec3(i) + 30);
      vec3 dice2 = pcg33(dice);

      t = mod(time.z + i * STEP2TIME, 16 * STEP2TIME);

      float i_env = exp(-t * exp2(5 + 5 * dice.y));

      float i_freq = exp2(5 + 10 * dice2.x);
      float phase = i_freq * t;

      vec2 i_wave = fbm32(vec2(exp2(4 - 4 * dice2.y) * orbit(phase) + exp2(-3 * dice2.z) * phase)).xy - 0.5;

      dest += 0.05 * sidechain * tanh(40 * i_env * i_wave);
    }
  }

  for(int i = 0; i < 64; i++)
  { // chords
    vec3 dice = pcg33(vec3(i));
    vec2 dicen = i_boxMuller(dice.xy);

    int i_frameOffset = int(exp2(9.2 * i_pluckOffset)) * i + SAMPLES_PER_STEP;
    int frameNote = max(0, frame - i_frameOffset) % (3 * SAMPLES_PER_STEP);
    int i_frameStart = frame - frameNote;
    int i_prog = max(0, i_frameStart / SAMPLES_PER_STEP - TENKAI_PROG_STEP) / 32 % 4 * 8;

    t = frameNote / SAMPLES_PER_SEC;
    float i_q = 3 * STEP2TIME - t;
    float i_env = smoothstep(0.0, 0.001, t) * smoothstep(0.0, 0.001, i_q) * exp(-t * 3.0);

    int chordn = 4 + clamp((i_frameStart / SAMPLES_PER_STEP - TENKAI_CHORD_START_STEP) / TENKAI_CHORD_LENGTH_STEP, 0, 4);
    float freq = i_p2f(54 + CHORDS[i % chordn + i_prog] + 0.1 * clamp(dicen.x, -1.0, 1.0));
    float phase = freq * time.w + dice.x;

    float i_k = 0.15 * i_pluckFilterEnv + 2 * t;
    vec2 i_wave = vec2(i_cheapFilterSaw(phase, i_k));
    dest += 0.03 * sidechain * i_env * i_wave * rot(i);
  }

  { // arp
    for(int i = 0; i < 64; i++)
    {
      int i_frameOffset = (i + TENKAI_ARP_START_STEP) * SAMPLES_PER_STEP;
      int frameNote = max(0, frame - i_frameOffset) % (64 * SAMPLES_PER_STEP);
      int i_frameStart = frame - frameNote;
      int i_prog = max(0, i_frameStart / SAMPLES_PER_STEP - TENKAI_PROG_STEP) / 32 % 4 * 8;

      t = frameNote / SAMPLES_PER_SEC;

      float t2f = mod(t, 3 * STEP2TIME);
      float i_t2i = floor(t / (3 * STEP2TIME));
      float i_q = 3 * STEP2TIME - t2f;
      float i_env = smoothstep(0.0, 0.002, t2f) * smoothstep(0.0, 0.01, i_q) * exp(-t2f * 12) * exp(-i_t2i);

      int arpseed = int(21 * fract(0.825 * i));
      float i_note = 42 + CHORDS[arpseed % 8 + i_prog] + arpseed / 8 * 12;
      float freq = i_p2f(i_note);
      vec2 phase = (t * freq + pcg33(vec3(i)).xy + 0.4 * fbm32(vec2(0.1 * t * freq)).xy);

      vec2 i_wave = vec2(sin(TAU * phase));
      dest += 0.2 * sidechain * i_env * i_wave * rot(time.w);
    }
  }

  waveOutSamples[frame] = i_masterAmp * tanh(dest);
}

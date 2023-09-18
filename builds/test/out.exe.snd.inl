// Generated with Shader Minifier 1.3.6 (https://github.com/laurentlb/Shader_Minifier/)
#ifndef SHADER_MINIFIER_IMPL
#ifndef SHADER_MINIFIER_HEADER
# define SHADER_MINIFIER_HEADER
# define VAR_g_waveOutPos "w"
#endif
#else // if SHADER_MINIFIER_IMPL
// sound_compute_shader.i
"#version 430\n"
 "layout(location=0) uniform int w;\n"
 "layout(std430,binding=0)buffer ssbo{vec2 s[];};layout(local_size_x=1)in;"
 "void main()"
 "{"
   "int g=int(gl_GlobalInvocationID.x)+w;"
   "vec2 l=vec2(float(g*256%65536-32768)/32768.,float(g*256%65536-32768)/32768.)*.1*exp(-float(g)*1e-4);"
   "l=clamp(l,-1.,1.);"
   "s[g]=l;"
 "}" 
#endif


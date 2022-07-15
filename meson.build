project('kms-quads',
  'c',
  default_options: [
    'warning_level=2',
    'c_std=c11',
  ]
)

cc = meson.get_compiler('c')

if cc.links('''
#include <stdint.h>
int main(void) {
  uint32_t x32 = 0;
  uint64_t x64 = 0;
  __atomic_load_n(&x32, __ATOMIC_SEQ_CST);
  __atomic_load_n(&x64, __ATOMIC_SEQ_CST);
  return 0;
}''', name : 'built-in atomics')
	libatomic = null_dep
else
	libatomic = cc.find_library('atomic')
endif

add_project_arguments(
  cc.get_supported_arguments([
    '-Wno-unused',
  ]),
  language: 'c'
)

defines = ['-D_POSIX_C_SOURCE=200809L']
if get_option('buildtype') == 'debug' or get_option('buildtype') == 'debugoptimized'
  defines += '-DDEBUG'
endif

deps = [
  dependency('libdrm'),
  dependency('gbm'),
  dependency('vulkan'),
  libatomic,
  cc.find_library('m')
]

# generate vulkan shaders
shaders = []
glslang = find_program('glslangValidator')
shaders_src = [
	'vulkan.vert',
	'vulkan.frag',
  'vkcube.vert',
  'vkcube.frag',
  'simple.vert',
  'simple.frag'
]

foreach shader : shaders_src
	name = shader.underscorify() + '_data'
	args = [glslang, '-V', '@INPUT@', '-o', '@OUTPUT@', '--vn', name]
	header = custom_target(
		shader + '_spv',
		output: shader + '.h',
		input: shader,
		command: args)
	shaders += [header]
endforeach

src = [
  'vulkan2.c',
  'modesetting.c',
  'esTransform.c',
  shaders,
]
executable('kms-quads', src,
  dependencies: deps,
  c_args: defines,
)

#version 450

layout(location = 0) in vec2 v_tex;

layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
	f_color = vec4(1,1,1, texture(tex, v_tex).r);
}

#version 430

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 uv;

void main() {
    uv = in_texcoord_0;
    gl_Position = vec4(in_position, 1.0);
}


#elif defined FRAGMENT_SHADER


uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;

void main() {
    fragColor = texture(tex, uv);
}
#endif
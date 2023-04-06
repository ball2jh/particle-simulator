#shader vertex
#version 330 core

layout(location = 0) in vec4 vertex_position;

void main() {
    gl_Position = vertex_position;
};

#shader fragment
#version 330 core
uniform vec2 iResolution;
in vec2 fragCoord;
out vec4 fragColor;

void main()
{
    vec3 circleColor = vec3(0.85, 0.35, 0.2);
    vec2 uv = (fragCoord.xy / iResolution.xy - 0.5) * 2.0;
    float distance = length(uv);

    vec4 color = vec4(distance < 1.0 ? vec3(1.0) : vec3(0.0), 1.0);
    if (distance > 1.0) {
        color.a = 0.0;
    }

    fragColor = color * vec4(circleColor, 1.0);
}
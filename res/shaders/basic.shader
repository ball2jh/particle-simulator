#shader vertex
#version 330 core

layout(location = 0) in vec4 vertex_position;

void main() {
    gl_Position = vertex_position;
};

#shader fragment
#version 330 core
out vec4 fragColor;

void main()
{
    fragColor = vec4(0.0, 1.0, 0.0, 1.0);
}
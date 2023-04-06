#include "renderer_2d.h"

#include <iostream>
#include <fstream>
#include <sstream>

Renderer2D::Renderer2D() : vertex_buffer(0), shader_program(0) {
}

Renderer2D::~Renderer2D() {
    glDeleteBuffers(1, &vertex_buffer);
    glDeleteProgram(shader_program);
}

bool Renderer2D::init(int *argc, char **argv) {
    ShaderProgramSource source = parseShader("res/shaders/Basic.shader");
    shader_program = createShader(source.vertex_source, source.fragment_source);
    glUseProgram(shader_program);

    glutInit(argc, argv);
    glutInitWindowSize(1024, 768);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutCreateWindow("Particle Simulator");
    glutDisplayFunc(display);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "GLEW initialization failed: %s\n", glewGetErrorString(err));
        return false;
    }

    return true;
}

void Renderer2D::createVBO(struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags) {
    // create buffer object
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

    float vertices[] = {
        -0.5f, -0.5f,
         0.5f, -0.5f,
         0.5f,  0.5f,
        -0.5f,  0.5f
    };
    // initialize buffer object
    unsigned int size = 8 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    std::string vertexShaderSource = R"(
    #version 330 core

    layout(location = 0) in vec4 vertex_position;

    void main() {
        gl_Position = vertex_position;
    };
    )";

    std::string fragmentShaderSource = R"(
        #version 330 core
        uniform vec2 iResolution;
        out vec4 fragColor;

        void main()
        {
            vec3 circleColor = vec3(0.85, 0.35, 0.2);
            vec2 uv = (gl_FragCoord.xy / iResolution.xy - 0.5) * 2.0;
            float distance = length(uv);

            vec4 color = vec4(distance < 1.0 ? vec3(1.0) : vec3(0.0), 1.0);
            if (distance > 1.0) {
                color.a = 0.0;
            }

            fragColor = color * vec4(circleColor, 1.0);
        }
    )";

    GLuint shader = createShader(vertexShaderSource, fragmentShaderSource);
    // ShaderProgramSource source = parseShader("res/shaders/basic.shader");
    // std::cout << "VERTEX" << std::endl;
    // std::cout << source.vertex_source << std::endl;
    // std::cout << "FRAGMENT" << std::endl;
    // std::cout << source.fragment_source << std::endl;

    //GLuint shader = createShader(source.vertex_source, source.fragment_source);
    //glUseProgram(shader);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(vbo_res, vertex_buffer, vbo_res_flags);
}

void Renderer2D::display() {
	glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Swap buffers
    glutSwapBuffers();
}

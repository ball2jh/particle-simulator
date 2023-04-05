#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "helpers/helper_cuda.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>

#define MAX_PARTICLES_PER_NODE 4
// vbo variables

GLuint vertex_buffer;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);

struct ShaderProgramSource {
    std::string vertex_source;
    std::string fragment_source;
};

static ShaderProgramSource parseShader(const std::string& file_path) {
    std::ifstream stream(file_path);

    enum class ShaderType {
        NONE = -1, VERTEX = 0, FRAGMENT = 1
    };

    std::string line;
    std::stringstream ss[2];
    ShaderType type = ShaderType::NONE;
    while (getline(stream, line)) {
        if (line.find("#shader") != std::string::npos) {
            if (line.find("vertex") != std::string::npos) {
                type = ShaderType::VERTEX;
            }
            else if (line.find("fragment") != std::string::npos) {
                type = ShaderType::FRAGMENT;
            }
        }
        else {
            ss[(int)type] << line << std::endl;
        }
    }
    return { ss[0].str(), ss[1].str() };
}

static GLuint compileShader(GLenum shader_type, const std::string& source) 
{
    GLuint id = glCreateShader(shader_type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE) {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char *error = (char*) alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, error);
        std::cout << "Failed to compile " << (shader_type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
        std::cout << error << std::endl;
        glDeleteShader(id);
        return 0;
    }
    
    return id;
}

static GLuint createShader(const std::string& vertexShader, const std::string& fragmentShader){
    GLuint program = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShader);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_POINTS, 0, 2);

    // Swap buffers
    glutSwapBuffers();
}

bool initGL(int *argc, char **argv)
{
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

void createVBO(GLuint *vertex_buffer, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags) {
    // create buffer object
    glGenBuffers(1, vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, *vertex_buffer);

    float vertices[] = {
        0.0f, 0.0f,
        5.0f, 5.0f
        };
    // initialize buffer object
    unsigned int size = 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    ShaderProgramSource source = parseShader("res/shaders/basic.shader");
    std::cout << "VERTEX" << std::endl;
    std::cout << source.vertex_source << std::endl;
    std::cout << "FRAGMENT" << std::endl;
    std::cout << source.fragment_source << std::endl;

    GLuint shader = createShader(source.vertex_source, source.fragment_source);
    glUseProgram(shader);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(vbo_res, *vertex_buffer, vbo_res_flags);
}

int main(int argc,  char** argv) {
    const int cuda_device = findCudaDevice(argc, (const char**)argv);
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));

    initGL(&argc, argv);

    createVBO(&vertex_buffer, &cuda_vbo_resource, 0);

    glutMainLoop();

    //glDeleteProgram(shader)

    return 0;
}
#include "shader.h"

#include <iostream>
#include <fstream>
#include <sstream>

Shader::Shader(const std::string& filepath) {
    std::pair<std::string, std::string> shaderSources = parseShader(filepath);
    vertex_src = shaderSources.first;
    fragment_src = shaderSources.second;
    create_program();
}

Shader::Shader(const std::string& vertex_src, const std::string& fragment_src) 
    : vertex_src(vertex_src), fragment_src(fragment_src) {
    create_program();
}

Shader::~Shader() {
    glDeleteProgram(shader_id);
}

void Shader::bind() const {
    glUseProgram(shader_id);
}

void Shader::unbind() const {
    glUseProgram(0);
}

std::pair<std::string, std::string> Shader::parseShader(const std::string& filepath) {
    std::ifstream file(filepath);

    if (!file.is_open()) {
		std::cout << "Failed to open file: " << filepath << std::endl;
		return { "", "" };
	}

    enum class ShaderType {
        NONE = -1, VERTEX = 0, FRAGMENT = 1
    };

    std::string line;
    std::stringstream ss[2];
    ShaderType type = ShaderType::NONE;

    while (getline(file, line)) {
        if (line.find("#shader") != std::string::npos) {
            if (line.find("vertex") != std::string::npos) {
                type = ShaderType::VERTEX;
            } else if (line.find("fragment") != std::string::npos) {
                type = ShaderType::FRAGMENT;
            }
        } else {
            ss[(int)type] << line << '\n';
        }
    }

    return { ss[0].str(), ss[1].str() };
}

void Shader::create_program() {
    const GLuint vertexShader = compile_shader(GL_VERTEX_SHADER, vertex_src.c_str());
    const GLuint fragmentShader = compile_shader(GL_FRAGMENT_SHADER, fragment_src.c_str());

    shader_id = glCreateProgram();
    glAttachShader(shader_id, vertexShader);
    glAttachShader(shader_id, fragmentShader);
    glLinkProgram(shader_id);
    glValidateProgram(shader_id);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

GLuint Shader::compile_shader(GLenum type, const GLchar* src) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        char* message = static_cast<char*>(alloca(length * sizeof(char)));
        glGetShaderInfoLog(shader, length, &length, message);
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}
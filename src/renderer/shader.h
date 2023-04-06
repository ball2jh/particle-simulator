#ifndef SHADER_H
#define SHADER_H

#pragma once

#include <string>
#include <GL/glew.h>

class Shader {
public:
    Shader(const std::string& filepath);
    Shader(const std::string& vertex_src, const std::string& fragment_src);
    ~Shader();

    void bind() const;
    void unbind() const;
private:
    std::pair<std::string, std::string> parseShader(const std::string& file_path);
    void create_program();
    GLuint compile_shader(GLenum shader_type, const char* source);
private:
    GLuint shader_id;
    std::string vertex_src;
    std::string fragment_src;
};


#endif // SHADER_H
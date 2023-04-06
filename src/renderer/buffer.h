#ifndef BUFFER_H
#define BUFFER_H

#pragma once

#include <GL/glew.h>

class VertexBuffer {
public:
    VertexBuffer(uint32_t size);
    ~VertexBuffer();

    void bind() const;
    void unbind() const;
    void set_data(const void* data, uint32_t size);
private:
	uint32_t buffer_id;
};

class IndexBuffer {
public:
    IndexBuffer(uint32_t* indices, uint32_t count);
    ~IndexBuffer();

    void bind() const;
    void unbind() const;
    uint32_t get_count() const;
private:
    uint32_t buffer_id;
    uint32_t count;
};

#endif // BUFFER_H
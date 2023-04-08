#ifndef VERTEX_BUFFER_H
#define VERTEX_BUFFER_H

#pragma once

#include "buffer.h"

class VertexBuffer : public Buffer {
public:
    VertexBuffer(uint32_t size);
    ~VertexBuffer();

    void bind() const override;
    void unbind() const override;
    void set_data(const void* data, uint32_t size);
};

#endif // VERTEX_BUFFER_H
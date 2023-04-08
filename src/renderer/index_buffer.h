#ifndef INDEX_BUFFER_H
#define INDEX_BUFFER_H

#pragma once

#include "buffer.h"

class IndexBuffer : public Buffer {
public:
    IndexBuffer(uint32_t* indices, uint32_t count);
    ~IndexBuffer();

    void bind() const override;
    void unbind() const override;
    uint32_t get_count() const;

private:
    uint32_t count;
};

#endif // INDEX_BUFFER_H
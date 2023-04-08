#ifndef BUFFER_H
#define BUFFER_H

#pragma once

#include <GL/glew.h>
#include <cstdint>

class Buffer {
public:
    virtual ~Buffer() = default;

    virtual void bind() const = 0;
    virtual void unbind() const = 0;

protected:
    uint32_t buffer_id;
};

#endif // BUFFER_H
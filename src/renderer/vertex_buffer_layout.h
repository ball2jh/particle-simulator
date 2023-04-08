#ifndef VERTEX_BUFFER_LAYOUT_H
#define VERTEX_BUFFER_LAYOUT_H

#include <vector>
#include "GL/glew.h"

class VertexBufferLayout {
public:
    struct Element {
        unsigned int size;
        unsigned int type;
        unsigned char normalized;

        Element(unsigned int size, unsigned int type, unsigned char normalized)
            : size(size), type(type), normalized(normalized) {}

        static unsigned int get_size_of_type(unsigned int type) {
            switch (type) {
                case GL_FLOAT:          return 4;
                case GL_UNSIGNED_INT:   return 4;
                case GL_UNSIGNED_BYTE:  return 1;
            }
            return 0;
        }
    };

private:
    std::vector<Element> elements_;
    unsigned int stride_;

public:
    VertexBufferLayout() : stride_(0) {}

    template<typename T>
    void push(unsigned int count);

    inline const std::vector<Element>& get_elements() const { return elements_; }
    inline unsigned int get_stride() const { return stride_; }
};

#endif // VERTEX_BUFFER_LAYOUT_H
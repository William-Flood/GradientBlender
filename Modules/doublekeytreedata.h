#ifndef doublekeytreedata
#define doublekeytreedata

#include <numpy/ndarrayobject.h>

typedef enum Color { 
    BLACK, 
    RED 
} Color;

typedef enum Direction { 
    LEFT, 
    RIGHT 
} Direction;


struct DoubleKeyTreeNode {
    npy_int32 y;
    npy_int32 x;
    struct DoubleKeyTreeNode *underLeaf;
    struct DoubleKeyTreeNode *overLeaf;
    struct DoubleKeyTreeNode *parent;
    Color color;
};

#endif
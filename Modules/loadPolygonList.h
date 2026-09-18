#ifndef findtrianglecut
#define findtrianglecut

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <doublekeytreedata.h>

struct DoubleKeyTreeNode* getVertexTree(PyObject *polygonList);
npy_intp getResultCount(struct DoubleKeyTreeNode* tree);
npy_int32* fillVertexList(struct DoubleKeyTreeNode* vertexTree, npy_int32* orderResults);
int ftc_init_numpy();


#endif
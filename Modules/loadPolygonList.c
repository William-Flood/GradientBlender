#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <stdbool.h>
#include "doublekeytreeops.h"
#include "loadPolygonList.h"

// Attempted solution found on StackOverflow; unsuccessful
int ftc_init_numpy(){
    return PyArray_ImportNumPyAPI();
}


struct DoubleKeyTreeNode* getVertexTree(PyObject *polygonList) {
    PyObject *polygoniter = PyObject_GetIter(polygonList);
    if (!polygoniter) {
        PyErr_SetString(PyExc_ValueError, "Expected an iterable");
        return NULL;
    }
    struct DoubleKeyTreeNode* tree = NULL;
    
    while (true) {
        PyObject *polygonView = PyIter_Next(polygoniter);
        if (!polygonView) {
            // nothing left in the iterator
            break;
        }
        if (!PyArray_Check(polygonView)) {
            Py_DecRef(polygoniter);
            Py_DecRef(polygonView);
            PyErr_SetString(PyExc_ValueError, "Non-array found inside argument");
            return NULL;
        }
        if (NPY_INT32 != PyArray_TYPE(polygonView)) {
            Py_DecRef(polygoniter);
            Py_DecRef(polygonView);
            PyErr_SetString(PyExc_ValueError, "Array not int32");
            return NULL;
        }
        if (!PyArray_IS_C_CONTIGUOUS(polygonView)) {
            Py_DecRef(polygoniter);
            Py_DecRef(polygonView);
            PyErr_SetString(PyExc_ValueError, "Array not contiguous");
            return NULL;
        }
        if (PyArray_NDIM(polygonView) != 2) {
            Py_DecRef(polygoniter);
            Py_DecRef(polygonView);
            PyErr_SetString(PyExc_ValueError, "Array must have 2 dimensions");
            return NULL;
        }
        if (PyArray_DIM(polygonView, 0 ) != 2) {
            Py_DecRef(polygoniter);
            Py_DecRef(polygonView);
            PyErr_SetString(PyExc_ValueError, "Array must have a size of 2 in dimension 0");
            return NULL;
        }


        const npy_int32 *intarray = (const npy_int32*)PyArray_DATA(polygonView);
        npy_intp pointCount = PyArray_DIM(polygonView, 1);
        npy_intp *strides_bytes = PyArray_STRIDES(polygonView);
        npy_intp strides[] = {strides_bytes[0] / sizeof(npy_int32), strides_bytes[1] / sizeof(npy_int32)};
        for (npy_intp inputPointIndex = 0; inputPointIndex < pointCount; inputPointIndex++) {
            npy_int32 y = intarray[strides[1] * inputPointIndex];
            npy_int32 x = intarray[strides[1] * inputPointIndex + strides[0]];
            if (insertToTree(
                y,
                x,
                &tree
            ) == -1) {
                chop(tree);
                Py_DecRef(polygoniter);
                Py_DecRef(polygonView);
                PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when ingesting point list");
                return NULL;
            }
        }
        Py_DecRef(polygonView);
    }
    Py_DecRef(polygoniter);
    return tree;
}


npy_intp getResultCount(struct DoubleKeyTreeNode* tree) {
    return getCount(tree);
}

npy_int32* fillVertexList(struct DoubleKeyTreeNode* vertexTree, npy_int32* orderResults) {
    getOrder(vertexTree, orderResults);
    chop(vertexTree);
    
    return orderResults;
}
#define PY_SSIZE_T_CLEAN
#include <pyconfig.h>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "doublekeytreedata.h"
#include "doublekeytreeops.h"
#include "loadPolygonList.h"
#include "point_orderer.h"

PyDoc_STRVAR(module_doc, "This contains supplemental c code for the gradient engine application.");

static PyObject * ge_foo(PyObject *module, PyObject *args)
{
    PyObject *polygonList;
    npy_int32 res = 0;
    if (!PyArg_ParseTuple(args, "O:foo", &polygonList)) {
        return NULL;
    }
    struct DoubleKeyTreeNode* vertexTree = getVertexTree(polygonList);
    if (vertexTree == NULL) {
        return NULL;
    }
    npy_intp resultCount = getResultCount(vertexTree);
    npy_intp resultShape[] = {resultCount, 2};
    PyArrayObject* results = PyArray_SimpleNew(2, resultShape, NPY_INT32);
    if (results == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when creating filtered point list");
        return NULL;
    }
    npy_int32* resultData = PyArray_DATA(results);
    fillVertexList(vertexTree, resultData);
    return results;
    
}

static PyObject * ge_order_border_points(PyObject *module, PyObject *args) {
    PyObject *borderPointsObj;
    PyObject *regionMatrixObj;
    if (!PyArg_ParseTuple(args, "OO:order_border_points", &borderPointsObj, &regionMatrixObj)) {
        return NULL;
    }
    if (!PyArray_Check(borderPointsObj)) {
        PyErr_SetString(PyExc_ValueError, "Non-array passed as border point list");
        return NULL;
    }
    if (!PyArray_Check(regionMatrixObj)) {
        PyErr_SetString(PyExc_ValueError, "Non-array passed as region values");
        return NULL;
    }
    PyArrayObject *borderPoints = (PyArrayObject *)borderPointsObj;
    PyArrayObject *regionMatrix = (PyArrayObject *)regionMatrixObj;
    return orderBorderPoints(borderPoints, regionMatrix);
}

static int ge_modexec(PyObject *m)
{
    int ftc_res = ftc_init_numpy();
    int pod_res = pod_init_numpy();
    int main_res = PyArray_ImportNumPyAPI();
    return (main_res > 0) && (pod_res > 0) && (ftc_res > 0);
}

static PyModuleDef_Slot ge_slots[] = {

    /* exec function to initialize the module (called as part of import
     * after the object was added to sys.modules)
     */
    {Py_mod_exec, ge_modexec},

    /* Signal that this module supports being loaded in multiple interpreters
     * with separate GILs (global interpreter locks).
     * See "Isolating Extension Modules" on how to prepare a module for this:
     *   https://docs.python.org/3/howto/isolating-extensions.html
     */
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},

    /* Signal that this module does not rely on the GIL for its own needs.
     * Without this slot, free-threaded builds of CPython will enable
     * the GIL when this module is loaded.
     */
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},

    {0, NULL}
};

PyDoc_STRVAR(ge_foo_doc,
"foo(arr)\n\nGiven a 2xn list of points, strips duplicates along the 1-axis and transposes");

PyDoc_STRVAR(ge_order_border_points_doc,
"ge_order_border_points(points, region_array)\n\nGiven a 2xn list of points and an array of region values, finds an ordering over the points");

static PyMethodDef ge_methods[] = {
    {"foo",                      ge_foo,                       METH_VARARGS,
        ge_foo_doc},
    {"order_border_points",       ge_order_border_points,      METH_VARARGS,
        ge_order_border_points_doc},
    {NULL,              NULL}           /* sentinel */
};

static struct PyModuleDef gemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "gradengc",
    .m_doc = module_doc,
    .m_size = 0,
    .m_methods = ge_methods,
    .m_slots = ge_slots
};

PyMODINIT_FUNC
PyInit_gradengc(void)
{
    return PyModuleDef_Init(&gemodule);
}
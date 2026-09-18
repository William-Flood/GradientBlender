#ifndef point_orderer
#define point_orderer

#include <numpy/ndarrayobject.h>

int pod_init_numpy();
PyArrayObject * orderBorderPoints(PyArrayObject * borderPoints, PyArrayObject * regionArray);

#endif
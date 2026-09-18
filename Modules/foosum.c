#include "numpy/ndarrayobject.h"
#include "foosum.h"

npy_int32 add_array(npy_int32 *intarray, npy_intp int_len) {
    npy_int32 res = 0;
    for (npy_intp i = 0; i < int_len; i++) {
            res = res + intarray[i];
    }
    return res;
}
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <stdbool.h>
#include <stdio.h>
#include <limits.h>
#include "point_orderer.h"


const npy_int32 neighborYOffsets[] = {1, 1, 0, -1, -1, -1, 0, 1};
const npy_int32 neighborXOffsets[] = {0, 1, 1, 1, 0, -1, -1, -1};


int pod_init_numpy() {
    // Called during module initialization to enable the numpy C api
    return PyArray_ImportNumPyAPI();
}

int validate2D_intArray(PyArrayObject * array) {
    // Ensures that the provided numpy array is 2-dimensional, contiguous, and of type np.int32
    if (array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Null value passed as argument");
        return -1;
    }
    if (NPY_INT32 != PyArray_TYPE(array)) {
        PyErr_SetString(PyExc_ValueError, "Array not int32");
        return -1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(array)) {
        PyErr_SetString(PyExc_ValueError, "Array not contiguous");
        return -1;
    }
    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must have 2 dimensions");
        return -1;
    }
    return 0;
}

int validateBorderPoints(PyArrayObject * borderPoints) {
    // Ensures that the provided list of border points is valid
    if (validate2D_intArray(borderPoints) < 0) {
        return -1;
    }
    if (PyArray_DIM(borderPoints, 0) != 2) {
        PyErr_SetString(PyExc_ValueError, "Point list must have a size of 2 in dimension 0");
        return -1;
    }
    if (PyArray_DIM(borderPoints, 1) == 0) {
        PyErr_SetString(PyExc_ValueError, "Point list cannot be empty");
        return -1;
    }
    if (PyArray_DIM(borderPoints, 1) > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "Point list exceeds maximum supported size");
        return -1;
    }
    return 0;
}


struct SortedPointResults {
    npy_intp pointCount;
    npy_int32 *pointList;
};

struct SortedPointResults *getSortedPoints(PyArrayObject * borderPoints) {
    // Given a numpy array of point coordinates, returns a sorted, deduplicated array of those coordinates
    npy_intp pointCount = PyArray_DIM(borderPoints, 1);
    npy_intp *stridesBytes = PyArray_STRIDES(borderPoints);
    struct DoubleKeyTreeNode* tree = NULL;
    const npy_int32 *intarray = (const npy_int32*)PyArray_DATA(borderPoints);
    npy_intp strides[] = {stridesBytes[0] / sizeof(npy_int32), stridesBytes[1] / sizeof(npy_int32)};
    for (npy_intp inputPointIndex = 0; inputPointIndex < pointCount; inputPointIndex++) {
        npy_int32 y = intarray[strides[1] * inputPointIndex];
        npy_int32 x = intarray[strides[1] * inputPointIndex + strides[0]];
        if (insertToTree(
            y,
            x,
            &tree
        ) == -1) {
            chop(tree);
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when ingesting point list");
            return NULL;
        }
    }
    struct SortedPointResults *results = malloc(sizeof(struct SortedPointResults));
    if (results == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when allocating point list results");
        chop(tree);
        return NULL;
    }
    results->pointCount = getCount(tree);
    // printf("point count: %i\n", results->pointCount);
    results->pointList = malloc(results->pointCount * 2 * sizeof(npy_int32));
    if (results->pointList == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when allocating allocating point list results");
        chop(tree);
        free(results);
        return NULL;
    }
    getOrder(tree, results->pointList);
    chop(tree);
    return results;
}


int findIndexIn(
    npy_int32 y, 
    npy_int32 x, 
    npy_int32 *pointList,
    npy_intp pointCount
) {
    // Finds the location of the specified coordinates within a sorted array of points
    npy_intp lowerBound = 0;
    npy_intp upperBound = pointCount - 1;
    // printf("Seeking %i %i\n", y, x);
    while (lowerBound <= (upperBound)) {
        npy_intp midpoint = (lowerBound + upperBound) / 2;
        // // printf("Range %i -> %i <- %i\n", lowerBound, midpoint, upperBound);
        // // printf("Range %i %i -> %i %i <- %i %i\n", 
        //     pointList[lowerBound * 2], pointList[lowerBound * 2 + 1], 
        //     pointList[midpoint * 2], pointList[midpoint * 2 + 1], 
        //     pointList[upperBound * 2], pointList[upperBound * 2 + 1]);
        if (pointList[midpoint * 2] ==y && pointList[midpoint * 2 + 1] ==x) {
            return midpoint;
        }
        else if (
            pointList[midpoint * 2] < y || 
            (pointList[midpoint * 2] ==y && pointList[midpoint * 2 + 1] < x) 
        ) {
            lowerBound = midpoint + 1;
        }
        else {
            upperBound = midpoint - 1;
        }
    }
    return -1;
}


npy_int32 getRegionValueAt(
    npy_int32 y, 
    npy_int32 x,
    npy_int32 *regionValues, 
    npy_intp regionHeight, 
    npy_intp regionWidth
) {
    // Gets the value of the original region matrix at the given coordinates
    // Defaults to -1 if the coordinates are out of bounds - this is consistent with
    // region values in the matrix being assigned from an enumerated list of
    // regions - ergo, every valid region will have a nonnegative associated value.
    if(y >= 0 && y < regionHeight && x >= 0 && x < regionWidth) {
        npy_int32 valueAt = regionValues[y * regionWidth + x];
        return valueAt;
    }
    else {
        return -1;
    }
}


int addCrossPointsOf(
    npy_int32 y, 
    npy_int32 x, 
    struct DoubleKeyTreeNode* tree
) {
    // Inserts the points directly above, directly below, and directly to the left and right of the provided coordinates
    if (insertToTree(
            y + 1,
            x,
            &tree
        ) == -1) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when ingesting cross point");
        return -1;
    }
    if (insertToTree(
            y - 1,
            x,
            &tree
        ) == -1) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when ingesting cross point");
        return -1;
    }
    if (insertToTree(
            y,
            x + 1,
            &tree
        ) == -1) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when ingesting cross point");
        return -1;
    }
    if (insertToTree(
            y,
            x - 1,
            &tree
        ) == -1) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when ingesting cross point");
        return -1;
    }
    return 0;
}


int checkNeighborConnection(
    npy_int32 y, 
    npy_int32 x, 
    struct SortedPointResults *borderPoints,
    npy_int32 *regionValues, 
    npy_intp regionHeight, 
    npy_intp regionWidth,
    bool isConnected[]
) {
    // Fills a boolean array with a marker indicating whether each point neighboring the provided coordinates
    // is a viable next point to walk to next.  The coordinates of each neighboring point follow the ordering
    // specified in neighborYOffsets and neighborXOffsets
    bool areNeighborsBorders[] = {false, false, false, false, false, false, false, false};
    bool areNeighborsInRegion[] = {true, true, true, true, true, true, true, true};
    npy_int32 pointRegion = getRegionValueAt(
        y, 
        x,
        regionValues, 
        regionHeight, 
        regionWidth
    );
    for (int i = 0; i < 8; i++) {
        npy_int32 neighborY = y + neighborYOffsets[i];
        npy_int32 neighborX = x + neighborXOffsets[i];
        // // printf("Checking neighbor %i %i\n", neighborY, neighborX);
        if (findIndexIn(neighborY, neighborX, borderPoints->pointList, borderPoints->pointCount) >= 0) {
            areNeighborsBorders[i] = true;
            // printf("Neighbor %i %i confirmed\n", neighborY, neighborX);
        }
        else{
            areNeighborsInRegion[i] = (pointRegion == getRegionValueAt(
                neighborY, 
                neighborX,
                regionValues, 
                regionHeight, 
                regionWidth
            ));
        }
    }
    bool areNeighborsConnectingExterior[8];
    for (int i = 0; i < 8; i++) {
        int cw_i = (i + 1) % 8;
        int ccw_i = (i + 7) % 8;
        areNeighborsConnectingExterior[i] = !areNeighborsInRegion[i] && (
            (i % 2 == 0) || 
            !(areNeighborsInRegion[cw_i] && areNeighborsInRegion[ccw_i])
        );
    }
    // printf("areNeighborsInRegion: [%i, %i, %i, %i, %i, %i, %i, %i]\n", 
    //     areNeighborsInRegion[0],
    //     areNeighborsInRegion[1],
    //     areNeighborsInRegion[2],
    //     areNeighborsInRegion[3],
    //     areNeighborsInRegion[4],
    //     areNeighborsInRegion[5],
    //     areNeighborsInRegion[6],
    //     areNeighborsInRegion[7]
    // );
    // printf("areNeighborsConnectingExterior: [%i, %i, %i, %i, %i, %i, %i, %i]\n", 
    //     areNeighborsConnectingExterior[0],
    //     areNeighborsConnectingExterior[1],
    //     areNeighborsConnectingExterior[2],
    //     areNeighborsConnectingExterior[3],
    //     areNeighborsConnectingExterior[4],
    //     areNeighborsConnectingExterior[5],
    //     areNeighborsConnectingExterior[6],
    //     areNeighborsConnectingExterior[7]
    // );
    for (int i = 0; i < 8; i++) {
        npy_int32 neighborY = y + neighborYOffsets[i];
        npy_int32 neighborX = x + neighborXOffsets[i];
        bool isBorder = areNeighborsBorders[i];
        bool isCorner = (i % 2) == 1;
        int cw_i = (i + 1) % 8;
        int ccw_i = (i + 7) % 8;
        // printf("areNeighborsConnectingExterior checks: %i - %i, %i - %i\n", 
        //     cw_i,
        //     areNeighborsConnectingExterior[cw_i], 
        //     ccw_i,
        //     areNeighborsConnectingExterior[ccw_i]
        // );
        bool hasConnectingExteriorAdjacency = areNeighborsConnectingExterior[cw_i] || areNeighborsConnectingExterior[ccw_i];
        bool hasConnectingInterior = areNeighborsInRegion[cw_i] || areNeighborsInRegion[ccw_i];
        bool connectsByInterior = (!isCorner || hasConnectingInterior);
        // printf("%i %i check: isBorder: %i; hasConnectingExteriorAdjacency: %i; connectsToInterior: %i\n", neighborY, neighborX, isBorder, hasConnectingExteriorAdjacency, connectsByInterior);
        isConnected[i] = isBorder && 
            hasConnectingExteriorAdjacency &&
            connectsByInterior;
    }
    return 0;
}

npy_intp addPointsToList(
    npy_int32 y, 
    npy_int32 x, 
    struct SortedPointResults *borderPoints,
    npy_int32 *regionValues, 
    npy_intp regionHeight, 
    npy_intp regionWidth,
    npy_int32 *resultsData,
    npy_intp startIndex,
    bool *searchedMask,
    npy_intp pointCount
) {
    //Walks along a set of points on the edge of a region of contiguous values, adding each to an array of points sequentially
    // starting from a given location
    npy_int32 navigateY = y;
    npy_int32 navigateX = x;
    // printf("Adding to walk at position %i from %i, %i\n", startIndex, y, x);
    bool continueNavigation = true;
    npy_intp insertIndex = startIndex;
    bool isConnected[] = {false, false, false, false, false, false, false, false};
    npy_intp initialPointIndex = findIndexIn(y, x, borderPoints->pointList, borderPoints->pointCount);
    if (initialPointIndex < 0) {
        PyErr_SetString(PyExc_ValueError, "Negative index found when searching neighbor point.  This likely indicates a bug in the traversal algorithm.");
        return -1;
    }
    searchedMask[initialPointIndex] = true;
    while (continueNavigation)
    {
        resultsData[insertIndex * 2] = navigateY;
        resultsData[insertIndex * 2 + 1] = navigateX;
        if (checkNeighborConnection(
            navigateY, 
            navigateX, 
            borderPoints,
            regionValues, 
            regionHeight, 
            regionWidth,
            isConnected
        ) < 0) {
            return -1;
        }
        continueNavigation = false;
        for (int i = 0; i < 8; i++) {
            if(isConnected[i]) {
                npy_int32 neighborY = navigateY + neighborYOffsets[i];
                npy_int32 neighborX = navigateX + neighborXOffsets[i];
                int neighborPointIndex = findIndexIn(neighborY, neighborX, borderPoints->pointList, borderPoints->pointCount);
                if (neighborPointIndex < 0) {
                    PyErr_SetString(PyExc_ValueError, "Negative index found when searching neighbor point.  This likely indicates a bug in the traversal algorithm.");
                    return -1;
                }
                if (!searchedMask[neighborPointIndex]) {
                    continueNavigation = true;
                    insertIndex += 1;
                    navigateY = neighborY;
                    navigateX = neighborX;
                    // printf("Stepping to %i %i\n", navigateY, navigateX);
                    searchedMask[neighborPointIndex] = true;
                    if (insertIndex >= pointCount) {
                        PyErr_SetString(PyExc_ValueError, "Point traversal overflow.  This likely indicates a bug in the traversal algorithm.");
                        return -1;
                    }
                    break;
                }
                else {
                    // printf("Skipping searched point %i %i\n", neighborY, neighborX);
                }
            }
        }
    }
    return insertIndex;
}


PyArrayObject *orderBorderPointsAfterExtraction(
    struct SortedPointResults *borderPoints,
    npy_int32 *regionValues, 
    npy_intp regionHeight, 
    npy_intp regionWidth
) {

    npy_intp resultShape[] = {borderPoints->pointCount, 2};
    PyArrayObject* results = PyArray_SimpleNew(2, resultShape, NPY_INT32);
    if (results == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when creating result array");
        return NULL;
    }
    npy_int32 *resultsData = PyArray_DATA(results);
    bool * searchedMask = calloc(borderPoints->pointCount, sizeof(bool));
    if (searchedMask == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed when creating searched mask");
        Py_DECREF(results);
        return NULL;
    }
    // The point list stores pairs of coordinates, thus, selecting the
    // point count (or closest even value) selects the point in the middle of the list
    npy_intp start_index = (borderPoints->pointCount) % 2 == 0 ? 
        borderPoints->pointCount :
        borderPoints->pointCount - 1;
    npy_int32 startY = borderPoints->pointList[start_index];
    npy_int32 startX = borderPoints->pointList[start_index + 1];
    // printf("Starting walk from %i %i", startY, startX);
    bool startConnections[] = {false, false, false, false, false, false, false, false};
    if (checkNeighborConnection(
        startY, 
        startX, 
        borderPoints,
        regionValues, 
        regionHeight, 
        regionWidth,
        startConnections
    ) < 0) {
        Py_DECREF(results);
        free(searchedMask);
        return NULL;
    }
    resultsData[0] = startY;
    resultsData[1] = startX;
    npy_intp startIndex = findIndexIn(startY, startX, borderPoints->pointList, borderPoints->pointCount);
    if (startIndex < 0) {
        PyErr_SetString(PyExc_ValueError, "Negative index found when searching start point.  This likely indicates a bug in the traversal algorithm");
        Py_DECREF(results);
        free(searchedMask);
        return NULL;
    }
    searchedMask[startIndex] = true;
    npy_intp lastInserted = 0;
    int passCount = 0;
    // printf("Looping through start connections");
    for (int i = 0; i < 8; i++) {
        if (startConnections[i] ) {
            npy_int32 y = startY + neighborYOffsets[i];
            npy_int32 x = startX + neighborXOffsets[i];
            npy_intp neighborIndex = findIndexIn(y, x, borderPoints->pointList, borderPoints->pointCount);
            if (neighborIndex < 0) {
                PyErr_SetString(PyExc_ValueError, "Negative index found when searching neighbor point.  This likely indicates a bug in the traversal algorithm");
                    Py_DECREF(results);
                    free(searchedMask);
                    return NULL;
            }
            if (!searchedMask[neighborIndex]) {
                if (passCount == 1) {
                    // printf("Reversing first %i elements\n", (lastInserted + 1));
                    for (int j = 0; j < (lastInserted + 1) / 2; j++) {
                        int swapTo = lastInserted - j;
                        npy_int32 tempY = resultsData[2 * swapTo];
                        npy_int32 tempX = resultsData[2 * swapTo + 1];
                        resultsData[2 * swapTo] = resultsData[2 * j];
                        resultsData[2 * swapTo + 1] = resultsData[2 * j + 1];
                        resultsData[2 * j] = tempY;
                        resultsData[2 * j + 1] = tempX;
                    }
                }
                else if (passCount > 1) {
                    PyErr_Format(PyExc_ValueError, "Too many branches found from start point %i %i - unable to determine traversal order", startY, startX);
                    Py_DECREF(results);
                    free(searchedMask);
                    return NULL;
                }
                lastInserted = addPointsToList(
                    y, 
                    x, 
                    borderPoints,
                    regionValues, 
                    regionHeight, 
                    regionWidth,
                    resultsData,
                    lastInserted + 1,
                    searchedMask,
                    borderPoints->pointCount
                );
                if (lastInserted == -1) {
                    Py_DECREF(results);
                    free(searchedMask);
                    return NULL;
                }
                passCount++;
                // break;
            }
        }
    }
    for (int i = lastInserted + 1; i < borderPoints->pointCount; i++) {
        resultsData[2 * i] = -1;
        resultsData[2 * i + 1] = -1;
    }
    free(searchedMask);
    return results;
}


PyArrayObject * orderBorderPoints(PyArrayObject * borderPoints, PyArrayObject * regionArray) {
    // Finds a walk for a set of points at the edge of a contiguous region within a matrix
    // Note: Returns an nx2 array, flipped from the expected input
    if (validateBorderPoints(borderPoints) < 0) {
        return NULL;
    }
    if (validate2D_intArray(regionArray) < 0) {
        return NULL;
    }
    struct SortedPointResults *pointOrderResults = getSortedPoints(borderPoints);
    if (pointOrderResults == NULL) {
        return NULL;
    }
    // printf("Order results count: %i", pointOrderResults->pointCount);
    const npy_int32 *regionValues = (const npy_int32*)PyArray_DATA(regionArray);
    const npy_intp *regionShape = PyArray_DIMS(regionArray);
    // printf("Walking through points\n");
    PyArrayObject * results = orderBorderPointsAfterExtraction(
        pointOrderResults,
        regionValues, 
        regionShape[0], 
        regionShape[1]
    );
    free(pointOrderResults->pointList);
    free(pointOrderResults);
    return results;
}
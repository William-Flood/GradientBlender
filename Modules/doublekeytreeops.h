#ifndef doublekeytreeops
#define doublekeytreeops

#include "doublekeytreedata.h"

int insertToTree(npy_int32 y, npy_int32 x, struct DoubleKeyTreeNode** tree);

void getOrder(struct DoubleKeyTreeNode* node, npy_int32* resultArray);

void chop(struct DoubleKeyTreeNode* node);

npy_intp getCount(struct DoubleKeyTreeNode* node);

#endif
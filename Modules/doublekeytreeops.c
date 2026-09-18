#include "doublekeytreeops.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct DoubleKeyTreeNode* ptnref;


bool continue_search(npy_int32 y, npy_int32 x, ptnref node) {
    if (node == NULL) {
        return false;
    }
    bool nodeMatches = node->x == x && node->y == y;
    bool nodeToLeft = node->y > y || (node->y == y && node->x > x);
    return !nodeMatches && 
        (
            nodeToLeft ? (node->underLeaf != NULL) : (node->overLeaf != NULL)
        );
}


ptnref getChild(ptnref node, Direction dir) {
    if(dir == LEFT) {
        return node->underLeaf;
    }
    else {
        return node->overLeaf;
    }
}


void setChild(ptnref node, Direction dir, ptnref new_child) {
    if(dir == LEFT) {
        node->underLeaf = new_child;
    }
    else {
        node->overLeaf = new_child;
    }
}

static Direction direction(const struct DoubleKeyTreeNode* N) {
    return N == N->parent->overLeaf ? RIGHT : LEFT;
}


ptnref rotate_subtree(struct DoubleKeyTreeNode** tree, ptnref sub, Direction dir) {
	ptnref sub_parent;
    sub_parent = sub->parent;
	ptnref new_root;
    new_root = getChild(sub, 1 - dir); // 1 - dir is the opposite direction
	ptnref new_child;
    new_child = getChild(new_root, dir);

	setChild(sub, 1 - dir, new_child);

	if (new_child) {
        new_child->parent = sub;
    }

	setChild(new_root, dir, sub);

	new_root->parent = sub_parent;
	sub->parent = new_root;
	if (sub_parent) {
		setChild(sub_parent, sub == sub_parent->overLeaf, new_root);
	} else {
		*tree = new_root;
    }

	return new_root;
}


int insertToTree(npy_int32 y, npy_int32 x, struct DoubleKeyTreeNode** tree){
    ptnref root;
    root = *tree;
    ptnref searchNode;
    searchNode = root;
    while(continue_search(y, x, searchNode)) {
        bool nodeToLeft = searchNode->y > y || (searchNode->y == y && searchNode->x > x);
        if (nodeToLeft) {
            searchNode = searchNode->underLeaf;
        }
        else {
            searchNode = searchNode->overLeaf;
        }
    }
    if (searchNode != NULL && (searchNode->y == y && searchNode->x == x)) {
        return 0;
    }
    
    ptnref node;
    node = malloc(sizeof(struct DoubleKeyTreeNode));
    if (node == NULL) {
        return -1;
    }
    node->y = y;
    node->x = x;
    node->underLeaf = NULL;
    node->overLeaf = NULL;
    node->parent = searchNode;
    node->color = RED;

	if (!searchNode) {
		*tree = node;
		return 0;
	}

    bool nodeToLeft = searchNode->y > y || (searchNode->y == y && searchNode->x > x);
    Direction dir = nodeToLeft ? LEFT : RIGHT;
	setChild(searchNode, dir, node);

	// rebalance the tree 
	do {
		// Case #1
		if (searchNode->color == BLACK) {
            return 0; 
        }

	    ptnref grandparent;
        grandparent = searchNode->parent;

		if (!grandparent) {
			// Case #4
			searchNode->color = BLACK;
			return 0;
		}

		dir = direction(searchNode);
		ptnref uncle;
        uncle = getChild(grandparent, 1 - dir);
		if (!uncle || uncle->color == BLACK) {
			if (node == getChild(searchNode, 1 - dir)) {
				// Case #5
				rotate_subtree(tree, searchNode, dir);
				node = searchNode;
				searchNode = getChild(grandparent, dir);
			}

			// Case #6
			rotate_subtree(tree, grandparent, 1 - dir);
			searchNode->color = BLACK;
			grandparent->color = RED;
			return 0;
		}
	
		// Case #2
		searchNode->color = BLACK;
		uncle->color = BLACK;
		grandparent->color = RED;
		node = grandparent;

	} while ((searchNode = node->parent));
    // This code is adapted from the red-black tree implementation provided in Wikipedia, which omits
    // the requirement stipulated by Cormen & al. that the root node remain black

	// Case #3
	return 0;
}

npy_intp getCount(struct DoubleKeyTreeNode* node) {
    npy_intp results = 0;
    if (node != NULL) {
        results += getCount(node->underLeaf);
        results += getCount(node->overLeaf);
        results += 1;
    }
    return results;
}


void fillResults(struct DoubleKeyTreeNode* node, npy_int32* resultsArray, npy_intp offset) {
    if (node!=NULL) {
        fillResults(node->underLeaf, resultsArray, offset);
        npy_intp offsetAfterUnderLeaf = getCount(node->underLeaf) + offset;
        resultsArray[2 * offsetAfterUnderLeaf] = node->y;
        resultsArray[2 * offsetAfterUnderLeaf + 1] = node->x;
        // printf("Adding %i %i at %i\n", node->y, node->x, offsetAfterUnderLeaf);
        fillResults(node->overLeaf, resultsArray, offsetAfterUnderLeaf + 1);
    }
}


void getOrder(struct DoubleKeyTreeNode* node, npy_int32* resultsArray) {
    fillResults(node, resultsArray, 0);
}

void chop(ptnref node) {
    if (node == NULL) {
        return;
    }
    if (node->underLeaf != NULL) {
        chop(node->underLeaf);
    }
    if (node->overLeaf != NULL) {
        chop(node->overLeaf);
    }
    free(node);
}

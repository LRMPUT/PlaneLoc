/*
    Copyright (c) 2017 Mobile Robots Laboratory at Poznan University of Technology:
    -Jan Wietrzykowski name.surname [at] put.poznan.pl

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#include "UnionFind.h"

UnionFind::UnionFind(int icount){
	set.resize(icount);
}

UnionFind::~UnionFind(){

}

int UnionFind::findSet(int node){
	if(set[node].parent == -1){
		return node;
	}

	set[node].parent = findSet(set[node].parent);
	return set[node].parent;
}

int UnionFind::unionSets(int node1, int node2){
	int node1Root = findSet(node1);
	int node2Root = findSet(node2);
	if(set[node1Root].rank > set[node2Root].rank){
		set[node2Root].parent = node1Root;
		set[node1Root].nsize += set[node2Root].nsize;
		return node1Root;
	}
	else if(set[node1Root].rank < set[node2Root].rank){
		set[node1Root].parent = node2Root;
		set[node2Root].nsize += set[node1Root].nsize;
		return node2Root;
	}
	else if(node1Root != node2Root){
		set[node2Root].parent = node1Root;
		set[node1Root].rank++;
		set[node1Root].nsize += set[node2Root].nsize;
		return node1Root;
	}
	return -1;
}

int UnionFind::size(int node){
	return set[findSet(node)].nsize;
}

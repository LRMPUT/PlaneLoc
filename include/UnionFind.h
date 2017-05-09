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

#ifndef UNIONFIND_H_
#define UNIONFIND_H_

#include <vector>
#include <cstddef>

/** \brief Struktura reprezentująca węzeł w klasie UnionFind.
 *
 */
struct SetNode{
	int parent, rank, nsize;
	SetNode() : parent(-1), rank(0), nsize(1) {}
	SetNode(int iparent, int irank, int insize) : parent(iparent), rank(irank), nsize(insize) {}
};

/** \brief Klasa reprezentująca rozłączne zbiory, umożliwiająca
 * 			efektywne ich łączenie.
 */
class UnionFind{
	std::vector<SetNode> set;
public:
	UnionFind(int icount);
	~UnionFind();

	/** \brief Funkcja znajdująca id zbioru, do którego należy węzeł node.
	 *
	 */
	int findSet(int node);

	/** \brief Funkcja łącząca dwa zbiory.
	 *
	 */
	int unionSets(int node1, int node2);

	/** \brief Funkcja zwracająca rozmiar zbioru.
	 *
	 */
	int size(int node);
};


#endif /* UNIONFIND_H_ */

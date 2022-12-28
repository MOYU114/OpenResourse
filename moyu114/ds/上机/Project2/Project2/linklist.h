#include<cstdio>
#include<iostream>
#include<malloc.h>
using namespace std;
typedef char ElemType;
typedef struct LNode {
	ElemType data;
	struct LNode* next;

}LinkNode;
void CreateNodeF(LinkNode*& L, ElemType a[], int n);
void CreateNodeR(LinkNode*& L, ElemType a[], int n);
void InitNode(LinkNode*& L);
void NodeDestroy(LinkNode*& L);
void NodePrint(LinkNode* L);
int NodeLength(LinkNode* L);
bool NodeEmpty(LinkNode* L);
bool NodeGet(LinkNode* L, int n, ElemType& result);
int NodeSearch(LinkNode* L, ElemType a);
bool NodeInsert(LinkNode*& L, int n, ElemType f);
bool NodeDelete(LinkNode*& L, int n, ElemType& e);
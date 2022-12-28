#include<cstdio>
#include<iostream>
#include<malloc.h>
using namespace std;
#define MAXSIZE 50
typedef int ElemType;
typedef struct {
	ElemType data[MAXSIZE];
	int length;

}SqList;
void CreateList(SqList*& L, ElemType a[], int n);
void InitList(SqList*& L);
void ListPrint(SqList* L);
int ListLength(SqList* L);
void ListDestroy(SqList*& L);
bool ListEmpty(SqList* L);
bool ListGet(SqList* L, int n, ElemType& result);
int ListSearch(SqList* L, ElemType a);
bool ListInsert(SqList*& L, int n, ElemType f);
bool ListDelete(SqList*& L, int n, ElemType& e);

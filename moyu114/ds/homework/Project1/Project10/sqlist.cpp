#include<cstdio>
#include<iostream>
#include<malloc.h>
using namespace std;
#define MAXSIZE 50
typedef char ElemType;
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
void CreateList(SqList* &L,ElemType a[],int n) {
	for (int i = 0;i < n;i++) {
		L->data[i]= a[i];
	}
	L->length = n;
}
void InitList(SqList*& L) {
	L = (SqList*)malloc(sizeof(SqList));
	L->length = 0;
}
void ListPrint(SqList* L) {
	for (int i = 0;i < L->length;i++)
		cout << L->data[i];
	cout << endl;
}
int ListLength(SqList* L) {
	return L->length;
}
void ListDestroy(SqList*& L) {
	free(L);
}
bool ListEmpty(SqList* L) {
	return (L->length==0);
}
bool ListGet(SqList* L,int n,ElemType & result) {
	if (n<1 || n>L->length)
		return false;
	result=L->data[n - 1];
	return true;
}
int ListSearch(SqList* L, ElemType a) {
	for (int i = 0;i < L->length;i++) {
		if (a == L->data[i])
			return i + 1;
		else
			return 0;
	}
}
bool ListInsert(SqList*& L, int n, ElemType f) {
	if (n<1||n>L->length+1) {
		return false;
	}
	n--;
	for (int i = L->length;i > n;i--) {
		L->data[i] = L->data[i - 1];
	}
	L->data[n] = f;
	L->length++;
	return true;
}
bool ListDelete(SqList*& L, int n, ElemType& e) {
	if (n<1 || n>L->length + 1) {
		return false;
	}
	n--;
	e = L->data[n];
	for (int i =n;i < L->length-1;i++) {
		L->data[i] = L->data[i +1];
	}
	L->length--;
	return true;
}
//int main() {
//	SqList* L;
//	ElemType e;
//	int locate;
//	cout << "初始化顺序表L" << endl;
//	InitList(L);
//	cout << "输入abcde" << endl;
//	ListInsert(L, 1, 'a');
//	ListInsert(L, 2, 'b');
//	ListInsert(L, 3, 'c');
//	ListInsert(L, 4, 'd');
//	ListInsert(L, 5, 'e');
//	cout << "输出L" << endl;;
//	ListPrint(L);
//	cout << "判断是否为空" << endl;
//	if (ListEmpty(L))
//		cout << "为空" << endl;
//	else
//		cout << "非空" << endl;
//	cout << "输出第三个数据" << endl;
//	if (ListGet(L, 3, e))
//		cout << e << endl;
//	else
//		cout << "不存在" << endl;
//	cout << "输出a的位置" << endl;
//	if (locate = ListSearch(L, 'a'))
//		cout << locate << endl;
//	else
//		cout << "不存在" << endl;
//	cout << "在第四个位置插入f" << endl;
//	ListInsert(L, 4, 'f');
//	cout << "输出顺序表" << endl;
//	ListPrint(L);
//	cout << "删除第三个元素" << endl;
//	ListDelete(L, 3, e);
//	cout << "输出顺序表" << endl;
//	ListPrint(L);
//	cout << "释放顺序表" << endl;
//	ListDestroy(L);
//
//}
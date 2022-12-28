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
//	cout << "��ʼ��˳���L" << endl;
//	InitList(L);
//	cout << "����abcde" << endl;
//	ListInsert(L, 1, 'a');
//	ListInsert(L, 2, 'b');
//	ListInsert(L, 3, 'c');
//	ListInsert(L, 4, 'd');
//	ListInsert(L, 5, 'e');
//	cout << "���L" << endl;;
//	ListPrint(L);
//	cout << "�ж��Ƿ�Ϊ��" << endl;
//	if (ListEmpty(L))
//		cout << "Ϊ��" << endl;
//	else
//		cout << "�ǿ�" << endl;
//	cout << "�������������" << endl;
//	if (ListGet(L, 3, e))
//		cout << e << endl;
//	else
//		cout << "������" << endl;
//	cout << "���a��λ��" << endl;
//	if (locate = ListSearch(L, 'a'))
//		cout << locate << endl;
//	else
//		cout << "������" << endl;
//	cout << "�ڵ��ĸ�λ�ò���f" << endl;
//	ListInsert(L, 4, 'f');
//	cout << "���˳���" << endl;
//	ListPrint(L);
//	cout << "ɾ��������Ԫ��" << endl;
//	ListDelete(L, 3, e);
//	cout << "���˳���" << endl;
//	ListPrint(L);
//	cout << "�ͷ�˳���" << endl;
//	ListDestroy(L);
//
//}
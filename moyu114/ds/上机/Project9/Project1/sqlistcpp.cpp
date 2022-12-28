#include"sqlist.h"
void CreateList(SqList*& L, ElemType a[], int n) {
	for (int i = 0;i < n;i++) {
		L->data[i] = a[i];
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
	return (L->length == 0);
}
bool ListGet(SqList* L, int n, ElemType& result) {
	if (n<1 || n>L->length)
		return false;
	result = L->data[n - 1];
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
	if (n<1 || n>L->length + 1) {
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
	for (int i = n;i < L->length - 1;i++) {
		L->data[i] = L->data[i + 1];
	}
	L->length--;
	return true;
}
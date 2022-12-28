#include<iostream>
#include<malloc.h>
typedef struct Node{
	int data;
	struct Node* next;
}List;
void InitList(List*& L) {
	L = (Node*)malloc(sizeof(Node));
	L->data = 0;
	L->next = NULL;
}
void CreateList(List*& L, int arr[],int n) {
	Node* p;
	L->data = arr[0];
	L->next = NULL;
	for (int i = 1;i < n;i++) {
		p = (Node*)malloc(sizeof(Node));
		p->data = arr[i];
		p->next = L->next;
		L->next = p;
	}
}

void InsertSort(List*& L) {
	List* DumNode;
	DumNode = (Node*)malloc(sizeof(Node));
	DumNode->next = L;
	List* temp, * p1 = L, * p2 = L->next;
	while (p2 != NULL) {

		if (p1->data <= p2->data) {//ÕýÐò
			p1 = p2;
			p2 = p2->next;
		}
		else {
			List* pre = DumNode;
			while (pre->next->data <= p2->data)
				pre = pre->next;
			p1->next = p2->next;
			p2->next = pre->next;
			pre->next = p2;

			p2 = p1->next;
		}
	}
	L = DumNode->next;
	free(DumNode);
	DumNode = NULL;
}
List* MergeSort(List* L1, List* L2) {
	List* DumNode;
	DumNode = (Node*)malloc(sizeof(Node));
	DumNode->next = NULL;
	List* p1 = L1, * p2 = L2, * p3 = DumNode, * crr;
	while (p1 != NULL && p2 != NULL) {
		if (p1->data > p2->data) {
			crr = (Node*)malloc(sizeof(Node));
			crr->data = p2->data;
			p3->next = crr;
			crr->next = NULL;
			p3 = p3->next;
			p2 = p2->next;
		}
		else if (p1->data <= p2->data) {
			crr = (Node*)malloc(sizeof(Node));
			crr->data = p1->data;
			p3->next = crr;
			crr->next = NULL;
			p3 = p3->next;
			p1 = p1->next;
		}
	}
	while (p1 != NULL || p2 != NULL)
	{
		if (p1 != NULL) {
			crr = (Node*)malloc(sizeof(Node));
			crr->data = p1->data;
			p3->next = crr;
			crr->next = NULL;
			p3 = p3->next;
			p1 = p1->next;
		}
		else {
			crr = (Node*)malloc(sizeof(Node));
			crr->data = p2->data;
			p3->next = crr;
			crr->next = NULL;
			p3 = p3->next;
			p2 = p2->next;
		}
		
	}return DumNode->next;
}
int main() {
	int arr[6] = { 2,5,4,3,4,1 };
	int arr2[6] = { 7,5,8,2,9,11 };
	List* L1,*L2,*L3,*p;
	InitList(L1);
	InitList(L2);
	CreateList(L1, arr, 6);
	CreateList(L2, arr2, 6);
	InsertSort(L1);
	InsertSort(L2);
	L3 = MergeSort(L1, L2);
	p = L3;
	for (int i = 0;i < 12;i++) {
		printf("%d", p->data);
		p = p->next;
	}
		
}
#include"linklist.h"
void fun(LinkNode* L1, LinkNode* L2, LinkNode*& L3) {
	LinkNode* p = L1->next, * q = L2->next, * r;
	L3 = L1;
	r = L3;
	free(L2);
	while (p != NULL && q != NULL) {
		r->next = p;r = p;p = p->next;
		r->next = q;r = q;q = q->next;
	}
	r->next = NULL;
	if (q != NULL) p = q;
	r->next = p;
}
int main() {
	LinkNode* L1, * L2, * L3;
	ElemType a[] = "abcdefgh";
	int n = 8;
	CreateNodeR(L1, a, n);
	cout << "L1:"<<endl;
	NodePrint(L1);
	ElemType b[] = "12345";
	n = 5;
	CreateNodeR(L2, b, n);
	cout << "L2:" << endl;
	NodePrint(L2);
	fun(L1, L2, L3);
	cout << "L3:" << endl;
	NodePrint(L3);
	NodeDestroy(L3);
}
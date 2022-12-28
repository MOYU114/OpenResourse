#include"linklist.h"
#include<string.h>
#define MAXSIZE 50
void add(LinkNode* L1, LinkNode* L2, LinkNode*& L3) {
	int temp = 0;
	LinkNode* p1 = L1->next, * p2 = L2->next, * p,* r;
	L3= (LinkNode*)malloc(sizeof LinkNode);
	r= L3;
	while (p1 != NULL && p2 != NULL) {
		p = (LinkNode*)malloc(sizeof LinkNode);
		p->data = ((p1->data-'0') + (p2->data - '0') +temp) % 10 + '0';
		temp = ((p1->data - '0') + (p2->data - '0') + temp) / 10;
		r->next = p;
		r = p;
		p1 = p1->next;
		p2 = p2->next;
	}
	if (p1 == NULL)p1 = p2;
	while (p1 != NULL) {
		p = (LinkNode*)malloc(sizeof LinkNode);
		p->data = ((p1->data - '0') + temp) % 10+'0';
		temp = ((p1->data - '0') + temp) / 10;
		r->next = p;
		r = p;
		p1 = p1->next;
	}
	if (temp > 0) {
		p = (LinkNode*)malloc(sizeof LinkNode);
		p->data ='0'+temp;
		r->next = p;
		r = p;
	}
	r->next = NULL;
}
void Reverse(LinkNode*& h) {
	LinkNode* p = h->next, * q;
	h->next = NULL;
	while (p != NULL) {
		q = p->next;
		p->next = h->next;
		h->next = p;
		p = q;
	}
}
int Mid(LinkNode*& h) {
	LinkNode* p = h, * q=h;
	int i = 1;
	while (i <= (NodeLength(h)+1) / 2) {
		p = p->next;
		i++;
	}
	return p->data-'0';
}
int main() {
	int i;
	char h1[MAXSIZE], h2[MAXSIZE];
	LinkNode* L1,* L2,* L3;
	cout << "输入第一个数：";
	scanf("%s", &h1);
	cout << "输入第二个数：";
	scanf("%s", &h2);
	CreateNodeF(L1, h1, strlen(h1));
	CreateNodeF(L2, h2, strlen(h2));
	add(L1, L2, L3);
	Reverse(L3);
	cout << "结果为：" ;
	NodePrint(L3);
	cout << endl;
	cout <<"中间值为：" <<Mid(L3);
	
}

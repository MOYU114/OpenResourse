#include"linklist.h"
void CreateNodeF(LinkNode*& L, ElemType a[], int n) {
	LinkNode* s;
	L=(LinkNode*)malloc(sizeof(LinkNode));
	L->next = NULL;
	for (int i = 0;i < n;i++) {
		s = (LinkNode*)malloc(sizeof(LinkNode));
		s->data = a[i];
		s->next = L->next;
		L->next = s;
	}
}
void CreateNodeR(LinkNode*& L, ElemType a[], int n) {
	LinkNode* s,*r;
	L = (LinkNode*)malloc(sizeof(LinkNode));
	L->next = NULL;
	r = L;
	for (int i = 0;i < n;i++) {
		s = (LinkNode*)malloc(sizeof(LinkNode));
		s->data = a[i];
		r->next = s;
		r = s;
	}
	r->next = NULL;
}
void InitNode(LinkNode*& L) {
	L = (LinkNode*)malloc(sizeof(LinkNode));
	L->next = NULL;
}
void NodeDestroy(LinkNode*& L) {
	LinkNode* temp = L, * loc = temp->next;
	while (loc != NULL) {
		free(temp);
		temp = loc;
		loc = temp->next;
	}
	free(temp);
}
void NodePrint(LinkNode* L) {
	LinkNode* s=L->next;
	while (s->next != NULL) {
		cout << s->data<<endl;
		s = s->next;
	}
	cout << s->data << endl;
}
int NodeLength(LinkNode* L) {
	int cnt = 0;
	LinkNode* s = L;
	while (s->next != NULL) {
		cnt++;
		s = s->next;
	}
	return cnt;
}

bool NodeEmpty(LinkNode* L) {
	return (L->next ==NULL);
}
bool NodeGet(LinkNode* L, int n, ElemType& result) {
	int i = 0;
	LinkNode* s = L;
	if (n < 0) return false;
	while (i < n && s->next != NULL) {
		i++;
		s = s->next;
	}
	if (s == NULL)
		return false;
	else
		result = s->data;
	return true;
	
}
int NodeSearch(LinkNode* L, ElemType a) {
	int i = 0;
	LinkNode* s = L;
	while (s->data!=a&&s->next != NULL) {
		s = s->next;
		i++;
	}
	if (s == NULL)
		return 0;
	else
		return i;

}
bool NodeInsert(LinkNode*& L, int n, ElemType f) {
	int i = 0;
	LinkNode* s = L,*p;
	if (n < 0) return false;
	while (i<n-1 && s->next != NULL) {
		s = s->next;
		i++;
	}
	if (s == NULL)
		return false;
	else {
		p = (LinkNode*)malloc(sizeof(LinkNode));
		p->data = f;
		p->next = s->next;
		s->next = p;
		return true;
	}

}
bool NodeDelete(LinkNode*& L, int n, ElemType& e) {
	int i = 0;
	LinkNode* s = L, * p;
	if (n < 0) return false;
	while (i < n - 1 && s->next != NULL) {
		s = s->next;
		i++;
	}
	if (s == NULL)
		return false;
	else {
		p=s->next;
		if (s == NULL)
			return false;
		e = p->data;
		s->next = p->next;
		free(p);
		return true;
	}

}
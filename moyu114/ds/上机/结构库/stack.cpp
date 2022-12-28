#include <iostream>
#include<malloc.h>
typedef char ElemType;
using namespace std;
typedef struct linknode {
	ElemType data;
	struct linknode* next;
}LinkNode;

void InitStack(LinkNode*& s) {
	s = (LinkNode*)malloc(sizeof(LinkNode));
	s->next = NULL;
}

bool StackEmpty(LinkNode* s) {
	return  (s->next == NULL);
}
void Push(LinkNode*& s, ElemType e) {
	LinkNode* p;
	p= (LinkNode*)malloc(sizeof(LinkNode));
	p->data = e;
	p->next = s->next;
	s->next = p;
}
bool Pop(LinkNode*& s, ElemType e) {
	LinkNode* p;
	if (s->next == NULL) {
		return false;
	}
	else {
		p = s->next;
		e = p->data;
		s->next = p->next;
		free(p);
		return true;
	}
}
bool GetTop(LinkNode* s, ElemType &e) {
	LinkNode* p;
	if (s->next == NULL) {
		return false;
	}
	else {
		p = s->next;
		e = p->data;

		return true;
	}
}
void DestoryStack(LinkNode *&s){
	LinkNode* p=s->next;
	while (p!= NULL) {
		free(s);
		s=p;
		p = p->next;
		
	}
}
int main() {
	LinkNode* s;
	ElemType e,e1='0';
	InitStack(s);
	StackEmpty(s) ? cout << "为空" << endl : cout << "不为空" << endl;
	Push(s, 'a');Push(s, 'b');Push(s, 'c');Push(s, 'd');Push(s, 'e');
	StackEmpty(s) ? cout << "为空" << endl : cout << "不为空" << endl;
	while(GetTop(s, e)){
		
		cout << e << endl;
		Pop(s, e1);
	}
	StackEmpty(s) ? cout << "为空" << endl : cout << "不为空" << endl;
	DestoryStack(s);
}
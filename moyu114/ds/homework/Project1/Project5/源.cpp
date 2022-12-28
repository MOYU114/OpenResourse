#include<cstdio>
#include<iostream>
#include<list>
#include<stack>
using namespace std;
//typedef struct {
//	int data;
//	LinkNode* next;
//}LinkNode;
//int fun(LinkNode* &p) {
//	LinkNode* q, * temp;
//	q = p->next;
//	p->next = NULL;
//	while (q != NULL) {
//		temp = q->next;
//		q->next = p->next;
//		p->next = q;
//		q=temp;
//	}
//}
//int fun(LinkNode*& ha, LinkNode*& hb, LinkNode*& hc) {
//	hc = ha;
//	LinkNode* p;
//	while (p->next != ha) {
//		p = p->next;
//	}
//	p->next = hb->next;
//
//	while (p->next != hb) {
//		p = p->next;
//	}
//	p->next = hc;
//	free(hb);
//}
//satck
bool Istrue(char cmd[],int n) 
{
	stack<char>st;
	int i=0;
	bool flag = true;
	while (flag && i < n)
	{
		if (cmd[i] == 'I') {
			st.push(cmd[i]);
		}
		else if (cmd[i] == 'O') {
			if(st.empty())
			{
				flag = false;
			}
			else
			{
				st.pop();
			}
		}
		else {
			flag = false;
		}

		i++;
	}
	if (st.empty()) {
	}
	else {
		flag = false;
	}
	return flag;
}
int main() {
	char a[8] = { 'I','I','I','O','O','O','O','O' };
	if (Istrue(a, 8)) {
		cout << "true";
	}
	else {
		cout << "false";
	}
}
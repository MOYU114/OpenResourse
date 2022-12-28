#include<iostream>
#include<malloc.h>
using namespace std;

typedef struct Node{
	int data;
	struct Node * next;
}QNode;
typedef struct {
	QNode* front, * rear;
}QuType;
void Destroyqueue(QuType*& qu) {
	QNode* pre, * p;
	pre = qu->front;
	if (pre != NULL) {
		p = pre->next;
		while (p != NULL) {
			free(pre);
			pre = p;p = p->next;
		}
		free(pre);
	}
	free(qu);
}
bool exist(QuType* qu,int no) {
	QNode* p=qu->front;
	bool find = false;
	while (p!=NULL&&!find) {
		if (p->data == no)find = true;
		else
			p = p->next;
	}
	return find;
}
void seeDoc() {
	int cmd, no;
	bool flag = true;
	QuType* qu;
	qu = (QuType*)malloc(sizeof(QuType));
	qu->front = qu->rear = NULL;
	while (flag) {
		back:
	cout << "���������ָ�1���Ŷ�2������3���鿴�Ŷ�4�������Ŷ�5���°ࣺ"<<endl;
	cin >> cmd;
	switch (cmd)
	{
		case 1:
			QNode* p;
			cout << "�����벡���ţ�";
			
			while (true) {
				cin >> no;
				if (exist(qu, no)) {
					cout << "�������ظ������������룺";
				}else{
					break;
			}

		}
			p = (QNode*)malloc(sizeof QNode);
			p->data = no;
			p->next = NULL;
			if (qu->rear == NULL) {
				qu->front = qu->rear = p;
			}
			else {
				qu->rear->next = p;
				qu->rear = p;
			}break;
		case 2:
			if (qu->front == NULL) {
				cout << "�޲��ˡ�" << endl;
			}
			else {
				p = qu->front;
				cout <<p->data << "�Ų�������" << endl;
				if (qu->rear == p) {
					qu->front = qu->rear = NULL;
				}else{
					qu->front = p->next;
				}
					
				free(p);
			}
			break;
		case 3:
			if (qu->front == NULL) {
				cout << "�޲��ˡ�" << endl;
			}
			else {
				p = qu->front;
				cout <<"�����еĲ��ˣ�" << endl;
				while (p!=NULL) {
					cout << p->data << endl;
					p = p->next;
				}
				cout << endl;
			}
			break;
		case 4:
			if (qu->front == NULL) {
				cout << "�޲��ˡ�" << endl;
			}
			else {
				p = qu->front;
				cout << "�벡�˰�����˳����";
				while (p != NULL) {
					cout << p->data << " " ;
					p = p->next;
				}
				cout << endl;	
			}
			Destroyqueue(qu);
			flag = false;
			break;

		case 5:
			if (qu->front != NULL)
				cout << "��֮��Ĳ�������������";
			flag = false;
			Destroyqueue(qu);
		break;
	default:
		cout << "������������������" << endl;
		goto back;
	}
}
}
int main() {
	seeDoc();
}
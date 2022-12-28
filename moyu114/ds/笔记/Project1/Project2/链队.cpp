#include<iostream>
using namespace std;
typedef char ElemType;
//�������н������DataNode����
typedef struct gnode{
	ElemType data;
	struct gnode* next;
}DataNode;
//������ͷ�������LinkQuNode
typedef struct {
	DataNode* front;
	DataNode* rear;
}LinkQuNode;

void InitQueue(LinkQuNode*& q) {
	q = (LinkQuNode*)malloc(sizeof(LinkQuNode));
	q->front = q->rear = NULL;
}

void DestoryQueue(LinkQuNode*& q) {
	DataNode* p = q->front, * r;
	if (p != NULL) {             //pָ���ͷ���ݽ��
		r = p->next;
		while (r != NULL) {     //�ͷ����ݽ��ռ�ÿռ�
			free(p);
			p = r;
			r = p->next;
		}
		free(p);              //�ͷ����һ�����ݽ��
	}
	free(q);                  //�ͷ����ӽ��ռ�ÿռ�
}

bool QueueEmpty(LinkQuNode* q)
{
	return(q->rear == NULL);
}

void enQueue(LinkQuNode*& q, ElemType e) {
	DataNode* p;
	p = (DataNode*)malloc(sizeof(DataNode));
	p->data = e;
	p->next = NULL;
	if (q->rear == NULL) {
		q->front = q->rear = p;//�������Ϊ�գ���ͷָ���βָ�붼ָ��p���½���Ƕ��׽�����Ƕ�β���
	}
	else {
		q->rear->next = p;   //��*p���������β������rearָ����
		q->rear = p;
	}
}
bool deQueue(LinkQuNode*& q,ElemType &e)
{
	DataNode* t;
	if (q->rear == NULL) return false;	//����Ϊ��
	t = q->front;		   		//tָ���һ�����ݽ��
	if (q->front == q->rear)  		//������ֻ��һ�����ʱ
		q->front = q->rear = NULL;
	else			   		//�������ж�����ʱ
		q->front = q->front->next;
	e = t->data;
	free(t);
	return true;
}
bool GetFront(LinkQuNode*& q, ElemType& e) {
	DataNode* t;
	if (q->rear == NULL) return false;	//����Ϊ��
	t = q->front;		   		//tָ���һ�����ݽ��
	e = t->data;
	return true;
}
int main() {
	LinkQuNode* s;
	ElemType e, e1 = '0';
	InitQueue(s);
	QueueEmpty(s) ? cout << "Ϊ��" << endl : cout << "��Ϊ��" << endl;
	enQueue(s, 'a');enQueue(s, 'b');enQueue(s, 'c');enQueue(s, 'd');enQueue(s, 'e');
	QueueEmpty(s) ? cout << "Ϊ��" << endl : cout << "��Ϊ��" << endl;
	while (GetFront(s, e)) {

		cout << e << endl;
		deQueue(s, e1);
	}
	QueueEmpty(s) ? cout << "Ϊ��" << endl : cout << "��Ϊ��" << endl;
	DestoryQueue(s);
}

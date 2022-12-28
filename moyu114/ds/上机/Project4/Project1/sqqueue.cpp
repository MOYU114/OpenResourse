#include <iostream>
#include<malloc.h>
using namespace std;
typedef char ElemType;
#define MaxSize 100
typedef struct
{
    ElemType data[MaxSize];
    int front, rear;
}SqQueue;
void InitQueue(SqQueue*& q) {
	q = (SqQueue*)malloc(sizeof SqQueue);
	q->front = q->rear = 0;
}
void DestoryQueue(SqQueue*& q) {
	free(q);
}
bool QueueEmpty(SqQueue* q) {
	return(q->front == q->rear);
}
bool enQueue(SqQueue*& q, ElemType e) {
	if ((q->rear+1)% MaxSize==q->front)return false;
	q->rear = (q->rear + 1) % MaxSize;
	q->data[q->rear] = e;
	return true;
}
bool deQueue(SqQueue*& q, ElemType& e) {
	if (q->front == q->rear)return false;
	q->front = (q->front + 1) % MaxSize;
	e = q->data[q->front];
	return true;
}

int main() {
	ElemType e;
	SqQueue* q;
	cout << "��ʼ������" << endl;
	InitQueue(q);
	cout << "a��b��c���" << endl;
	if(!enQueue(q, 'a'))cout<<"�������������ܽ���"<<endl;
	if (!enQueue(q, 'b'))cout << "�������������ܽ���" << endl;
	if (!enQueue(q, 'c'))cout << "�������������ܽ���" << endl;
	cout << "����һ��Ԫ��" << endl;
	if (!deQueue(q, e)) { 
		cout << "����Ϊ�գ����ܳ���" << endl; 
	}
	else {
		cout << e << endl;
	}
	cout << "d��e��f���" << endl;
	if (!enQueue(q, 'd'))cout << "�������������ܽ���" << endl;
	if (!enQueue(q, 'e'))cout << "�������������ܽ���" << endl;
	if (!enQueue(q, 'f'))cout << "�������������ܽ���" << endl;
	cout << "��������Ԫ��" << endl;
	while (!QueueEmpty(q)) {
		deQueue(q, e);
		cout << e << endl;
	}
	cout << "���ٶ���" << endl;
	DestoryQueue(q);
}
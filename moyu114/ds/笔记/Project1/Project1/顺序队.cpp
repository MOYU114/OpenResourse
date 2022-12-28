#include<iostream>
using namespace std;
typedef char ElemType;
#define MAXSIZE 50
typedef struct {
	ElemType data[MAXSIZE];
	int front, rear;
}SqQueue;

void InitQueue(SqQueue*& q) {
	q = (SqQueue*)malloc(sizeof SqQueue);
	q->front = q->rear = -1;
}
void DestoryQueue(SqQueue*& q) {
	free(q);
}
bool QueueEmpty(SqQueue* q) {
	return(q->front == q->rear);
}
bool enQueue(SqQueue*& q,ElemType e) {
	if (q->rear == MAXSIZE - 1)return false;
	q->rear++;
	q->data[q->rear] = e;
	return true;
}
bool deQueue(SqQueue*& q, ElemType& e) {
	if (q->front == q->rear)return false;
	q->front++;
	e = q->data[q->front];
	return true;
}
bool GetFront(SqQueue*& q, ElemType &e) {
	if (q->front == q->rear)return false;
	e = q->data[q->front];
	return true;
}
int main() {
	SqQueue* s;
	ElemType e, e1 = '0';
	InitQueue(s);
	QueueEmpty(s) ? cout << "为空" << endl : cout << "不为空" << endl;
	enQueue(s, 'a');enQueue(s, 'b');enQueue(s, 'c');enQueue(s, 'd');enQueue(s, 'e');
	QueueEmpty(s) ? cout << "为空" << endl : cout << "不为空" << endl;
	while (GetFront(s, e)) {

		cout << e << endl;
		deQueue(s, e1);
	}
	QueueEmpty(s) ? cout << "为空" << endl : cout << "不为空" << endl;
	DestoryQueue(s);
}

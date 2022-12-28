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
	cout << "初始化队列" << endl;
	InitQueue(q);
	cout << "a、b、c入队" << endl;
	if(!enQueue(q, 'a'))cout<<"队列已满，不能进队"<<endl;
	if (!enQueue(q, 'b'))cout << "队列已满，不能进队" << endl;
	if (!enQueue(q, 'c'))cout << "队列已满，不能进队" << endl;
	cout << "出队一个元素" << endl;
	if (!deQueue(q, e)) { 
		cout << "队列为空，不能出队" << endl; 
	}
	else {
		cout << e << endl;
	}
	cout << "d、e、f入队" << endl;
	if (!enQueue(q, 'd'))cout << "队列已满，不能进队" << endl;
	if (!enQueue(q, 'e'))cout << "队列已满，不能进队" << endl;
	if (!enQueue(q, 'f'))cout << "队列已满，不能进队" << endl;
	cout << "出队所有元素" << endl;
	while (!QueueEmpty(q)) {
		deQueue(q, e);
		cout << e << endl;
	}
	cout << "销毁队列" << endl;
	DestoryQueue(q);
}
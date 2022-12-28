#include"queue.h"
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
bool enQueue(SqQueue*& q, ElemType i, ElemType j) {
	if ((q->rear + 1) % MaxSize == q->front)return false;
	q->rear = (q->rear + 1) % MaxSize;
	q->i[q->rear] = i;q->j[q->rear] =j;
	return true;
}
bool deQueue(SqQueue*& q, ElemType& i, ElemType& j) {
	if (q->front == q->rear)return false;
	q->front = (q->front + 1) % MaxSize;
	i = q->i[q->front];j = q->j[q->front];
	return true;
}
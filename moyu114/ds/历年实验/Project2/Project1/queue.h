#include <iostream>
#include<malloc.h>
typedef int ElemType;
#define MaxSize 100
typedef struct
{
	ElemType i[MaxSize];
	ElemType j[MaxSize];
	int front, rear;
}SqQueue;
void InitQueue(SqQueue*& q);
void DestoryQueue(SqQueue*& q);
bool QueueEmpty(SqQueue* q);
bool enQueue(SqQueue*& q, ElemType i, ElemType j);
bool deQueue(SqQueue*& q, ElemType& i, ElemType& j);
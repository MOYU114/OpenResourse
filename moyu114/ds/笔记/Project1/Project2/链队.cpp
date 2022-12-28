#include<iostream>
using namespace std;
typedef char ElemType;
//单链表中结点类型DataNode定义
typedef struct gnode{
	ElemType data;
	struct gnode* next;
}DataNode;
//链队中头结点类型LinkQuNode
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
	if (p != NULL) {             //p指向队头数据结点
		r = p->next;
		while (r != NULL) {     //释放数据结点占用空间
			free(p);
			p = r;
			r = p->next;
		}
		free(p);              //释放最后一个数据结点
	}
	free(q);                  //释放链队结点占用空间
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
		q->front = q->rear = p;//如果队列为空，将头指针和尾指针都指向p，新结点是队首结点又是队尾结点
	}
	else {
		q->rear->next = p;   //将*p结点链到队尾，并将rear指向它
		q->rear = p;
	}
}
bool deQueue(LinkQuNode*& q,ElemType &e)
{
	DataNode* t;
	if (q->rear == NULL) return false;	//队列为空
	t = q->front;		   		//t指向第一个数据结点
	if (q->front == q->rear)  		//队列中只有一个结点时
		q->front = q->rear = NULL;
	else			   		//队列中有多个结点时
		q->front = q->front->next;
	e = t->data;
	free(t);
	return true;
}
bool GetFront(LinkQuNode*& q, ElemType& e) {
	DataNode* t;
	if (q->rear == NULL) return false;	//队列为空
	t = q->front;		   		//t指向第一个数据结点
	e = t->data;
	return true;
}
int main() {
	LinkQuNode* s;
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

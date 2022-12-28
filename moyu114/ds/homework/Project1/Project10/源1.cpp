#include "sqlist.cpp" //顺序表基本运算算法
#include <malloc.h>
#define Max 3
#define MAXL 10
typedef struct {
	int key;
	char data;
}RecType;
typedef struct node
{
	RecType Rec;
	struct node* next;
} NodeType;
void RadixSort1(RecType R[], int n)
{
	NodeType* head[Max], * tail[Max], * p, * t; //定义各链队的首尾指针
	int i, k;
	for (i = 0;i < Max;i++) //初始化各链队首、尾指针
		head[i] = tail[i] = NULL;
	for (i = 0;i < n;i++)
	{
		p = (NodeType*)malloc(sizeof(NodeType));//创建新节点
		p->Rec = R[i];
		p->next = NULL;
		k = R[i].key; //找第k个链队,k=0,1或2
		if (head[k] == NULL) //进行分配,采用前插法建表
		{
			head[k] = p; tail[k] = p;
		}
		else
		{
			tail[k]->next = p; tail[k] = p;
		}
	}
	p = NULL;
	for (i = 0;i < Max;i++) //对于每一个链队进行循环收集
		if (head[i] != NULL) //产生以p为首节点指针的单链表
		{
			if (p == NULL)
			{
				p = head[i]; t = tail[i];
			}
			else
			{
				t->next = head[i]; t = tail[i];
			}
		}
	i = 0;
	while (p != NULL) //将排序后的结果放到R[]数组中
	{
		R[i++] = p->Rec;
		p = p->next;
	}
}
int main()
{
	int i, n = 5;
	RecType R[MAXL] = { {1,'A'},{0,'B'},{0,'C'},{2,'D'},{1,'F'} };
	printf("排序前:\n ");
	for (i = 0;i < n;i++)
		printf("[%d,%c] ", R[i].key, R[i].data);
	printf("\n");
	RadixSort1(R, n);
	printf("排序后:\n ");
	for (i = 0;i < n;i++)
		printf("[%d,%c] ", R[i].key, R[i].data);
	printf("\n");
	return 1;
}

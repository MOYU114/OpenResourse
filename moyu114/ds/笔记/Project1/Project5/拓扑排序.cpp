#include<iostream>
#define MAXV 50
using namespace std;

typedef char InfoType;
//�����������
typedef struct
{
	int no;
	InfoType info;
}VertexType;
//�ڽӾ���
typedef struct
{
	int edges[MAXV][MAXV];
	int n, e;
	VertexType vexs[MAXV];
}MatGraph;
//�ڽӱ�
typedef struct ANode
{
	int adjvex;			//�ñߵ��յ���
	struct ANode* nextarc;	//ָ����һ���ߵ�ָ��
	InfoType weight;		//�ñߵ�Ȩֵ����Ϣ
}  ArcNode;
typedef struct 	       	//��ͷ�������
{
	VertexType data;         	//������Ϣ
	int count;           	//��Ŷ������
	ArcNode* firstarc;   	//ָ���һ����
} VNode;

typedef struct
{
	VNode adjlist[MAXV];	//�ڽӱ�
	int n, e;			//ͼ�ж�����n�ͱ���e
} AdjGraph;

void TopSort(AdjGraph* G)	//���������㷨
{
	int i,j;
	int St[MAXV],top = -1;	//ջSt��ָ��Ϊtop
	ArcNode* p;
	for (i = 0;i < G->n;i++)		//����ó�ֵ0
		G->adjlist[i].count = 0;
	for (i = 0;i < G->n;i++)		//�����ж�������
	{
		p = G->adjlist[i].firstarc;
		while (p != NULL)
		{
			G->adjlist[p->adjvex].count++;
			p = p->nextarc;
		}
	}
	for (i = 0;i < G->n;i++)		//�����Ϊ0�Ķ����ջ
		if (G->adjlist[i].count == 0)
		{
			top++;
			St[top] = i;
		}
	while (top > -1)			//ջ����ѭ��
	{
		i = St[top];top--;			//��ջһ������i
		printf("%d ", i);		//����ö���
		p = G->adjlist[i].firstarc;		//�ҵ�һ���ڽӵ�
		while (p != NULL)		//������i�ĳ����ڽӵ����ȼ�1
		{
			j = p->adjvex;
			G->adjlist[j].count--;
			if (G->adjlist[j].count == 0)	//�����Ϊ0���ڽӵ��ջ
			{
				top++;
				St[top] = j;
			}
			p = p->nextarc;		//����һ���ڽӵ�
		}
	}
}
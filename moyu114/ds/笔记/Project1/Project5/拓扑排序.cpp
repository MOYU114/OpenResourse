#include<iostream>
#define MAXV 50
using namespace std;

typedef char InfoType;
//顶点基本类型
typedef struct
{
	int no;
	InfoType info;
}VertexType;
//邻接矩阵
typedef struct
{
	int edges[MAXV][MAXV];
	int n, e;
	VertexType vexs[MAXV];
}MatGraph;
//邻接表
typedef struct ANode
{
	int adjvex;			//该边的终点编号
	struct ANode* nextarc;	//指向下一条边的指针
	InfoType weight;		//该边的权值等信息
}  ArcNode;
typedef struct 	       	//表头结点类型
{
	VertexType data;         	//顶点信息
	int count;           	//存放顶点入度
	ArcNode* firstarc;   	//指向第一条边
} VNode;

typedef struct
{
	VNode adjlist[MAXV];	//邻接表
	int n, e;			//图中顶点数n和边数e
} AdjGraph;

void TopSort(AdjGraph* G)	//拓扑排序算法
{
	int i,j;
	int St[MAXV],top = -1;	//栈St的指针为top
	ArcNode* p;
	for (i = 0;i < G->n;i++)		//入度置初值0
		G->adjlist[i].count = 0;
	for (i = 0;i < G->n;i++)		//求所有顶点的入度
	{
		p = G->adjlist[i].firstarc;
		while (p != NULL)
		{
			G->adjlist[p->adjvex].count++;
			p = p->nextarc;
		}
	}
	for (i = 0;i < G->n;i++)		//将入度为0的顶点进栈
		if (G->adjlist[i].count == 0)
		{
			top++;
			St[top] = i;
		}
	while (top > -1)			//栈不空循环
	{
		i = St[top];top--;			//出栈一个顶点i
		printf("%d ", i);		//输出该顶点
		p = G->adjlist[i].firstarc;		//找第一个邻接点
		while (p != NULL)		//将顶点i的出边邻接点的入度减1
		{
			j = p->adjvex;
			G->adjlist[j].count--;
			if (G->adjlist[j].count == 0)	//将入度为0的邻接点进栈
			{
				top++;
				St[top] = j;
			}
			p = p->nextarc;		//找下一个邻接点
		}
	}
}
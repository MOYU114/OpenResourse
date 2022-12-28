#include<iostream>
#include<malloc.h>
using namespace std;
#define MAXV 50
#define INF 32767
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

typedef struct Vnode
{
	VertexType data;			//顶点信息
	ArcNode* firstarc;		//指向第一条边
}  VNode;

typedef struct
{
	VNode adjlist[MAXV];	//邻接表
	int n, e;			//图中顶点数n和边数e
} AdjGraph;

void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e) {
	int i, j=0;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof AdjGraph);
	for (i = 0;i < n;i++) {
		G->adjlist[i].firstarc = NULL;//给邻接表中所有头结点的指针域置初值
	}
	for (i = 0;i < n;i++) {
		for (j = n - 1;j >= 0;j--) {
			if (A[i][j] != 0 && A[i][j] != INF) { //存在一条边
				p = (ArcNode*)malloc(sizeof ArcNode);//创建一个结点p
				p->adjvex = j;                     //存放邻接点
				p->weight = A[i][j];              //存放权
				p->nextarc = G->adjlist[i].firstarc;//采用头插法插入结点p
				G->adjlist[i].firstarc = p;
			}
			
		}
		
	}
	G->n = n;G->e = e;
}
void DispAdj(AdjGraph* G) {
	int i;
	ArcNode* p;
	for (i = 0;i < G->n;i++) {
		p = G->adjlist[i].firstarc;
		printf("%3d: ", i);
		while (p != NULL) {
			printf("%3d[%d]-> ", p->adjvex, p->weight);
			p = p->nextarc;
		}
		printf("∧\n");
	}
}
void DestroyAdj(AdjGraph*& G) {
	int i;
	ArcNode* pre, * p;

	for (i = 0;i < G->n;i++) {//扫描所有的单链表
		pre = G->adjlist[i].firstarc;//p指向第i个单链表的首结点
		if (pre != NULL) {
			p = pre->nextarc;
			while (p != NULL)	//释放第i个单链表的所有边结点
			{
				free(pre);
				pre = p; p = p->nextarc;
			}
			free(pre);
		}
	}
	free(G);//释放头结点数组
}
void MatToList(MatGraph g, AdjGraph*& G) {
	//将邻接矩阵g转换成邻接表G
	int i, j;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof(AdjGraph));
	for (i = 0;i < G->n;i++) {//将邻接表中所有头结点的指针域置初值
		G->adjlist[i].firstarc = NULL;
	}
	for (i = 0;i < g.n;i++) {//检查邻接矩阵中每个元素
		for (j = g.n - 1;j >= 0;j--) {
			if (g.edges[i][j] != 0 && g.edges[i][j] != INF) {//存在一条边
				p = (ArcNode*)malloc(sizeof(ArcNode));//建一个边结点p
				p->adjvex = j;p->weight = g.edges[i][j];
				p->nextarc = G->adjlist[i].firstarc;;     //采用头插法插入结点p
				G->adjlist[i].firstarc = p;
			}
		}
	}
	G->n = g.n;G->e = g.e;
}
void ListToMat(AdjGraph* G, MatGraph& g)
//将邻接表G转换成邻接矩阵g
{
	int i;
	ArcNode* p;
	for (i = 0;i < G->n;i++)			//扫描所有的单链表
	{
		p = G->adjlist[i].firstarc;		//p指向第i个单链表的首结点
		while (p != NULL)		//扫描第i个单链表
		{
			g.edges[i][p->adjvex] = 1;//将vi中能够到达的路径置一
			p = p->nextarc;
		}
	}
	g.n = G->n; g.e = G->e;
}
int visited[MAXV];
void findpath(AdjGraph* G, int u, int v, int path[], int d, int length)
{ //d表示path中顶点个数，初始为0；length表示路径长度，初始为0
	int w, i;
	ArcNode* p;
	path[d] = u; d++; //顶点u加入到路径中，d增1
	visited[u] = 1; //置已访问标记
	if (u == v && d > 0) //找到一条路径则输出
	{
		printf(" 路径长度:%d, 路径:", length);
		for (i = 0;i < d;i++)
			printf("%2d", path[i]);
		printf("\n");
	}
	p = G->adjlist[u].firstarc; //p指向顶点u的第一个邻接点
	while (p != NULL)
	{
		w = p->adjvex; //w为顶点u的邻接点
		if (visited[w] == 0) //若w顶点未访问,递归访问它
			findpath(G, w, v, path, d, p->weight + length);
		p = p->nextarc; //p指向顶点u的下一个邻接点
	}
	visited[u] = 0;
}
	
int main() {
	AdjGraph* G;
	int A[MAXV][MAXV] = {
	{0,4,6,6,INF,INF,INF},
	{INF,0,1,INF,7,INF,INF},
	{INF,INF,0,INF,6,4,INF},
	{INF,INF,2,0,INF,5,INF},
	{INF,INF,INF,INF,0,INF,6},
	{INF,INF,INF,INF,1,0,8},
	{INF,INF,INF,INF,INF,INF,0} };

	int n = 7, e = 12;
	CreateAdj(G, A, n, e); //创建《教程》中图8.35的邻接表
	printf("图G的邻接表:\n");
	DispAdj(G); //输出邻接表
	int u = 0, v = 5;
	int path[MAXV];
	printf("从%d->%d的所有路径:\n", u, v);
	findpath(G, u, v, path, 0, 0);
	DestroyAdj(G);
	return 1;
}


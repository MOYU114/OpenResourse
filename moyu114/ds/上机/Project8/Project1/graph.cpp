#include<iostream>
#include<malloc.h>
using namespace std;
#define MAXV 100
typedef char InfoType;
#define INF 32767
//邻接表
typedef struct //顶点类型
{
	int no;
	InfoType info;
}VertexType;

typedef struct
{
	int edges[MAXV][MAXV];
	int n,e;
	VertexType vexs[MAXV];
}MatGraph;
//邻接矩阵
typedef struct ANode//声明边结点类型
{
	int adjvex;			//该边的终点编号
	struct ANode* nextarc;	//指向下一条边的指针
	int weight;		//该边的权值等信息
}  ArcNode;

typedef struct Vnode//声明表头结点类型
{
	VertexType data;			//顶点信息

	ArcNode* firstarc;		//指向第一条边
}  VNode;
typedef struct//声明图邻接表类型
{
	VNode adjlist[MAXV];	//邻接表
	int n,e;			//图中顶点数n和边数e
} AdjGraph;
//邻接矩阵
void CreateMat(MatGraph& g, int A[MAXV][MAXV], int n, int e) {
	int i, j;
	g.n = n;g.e = e;
	for (i = 0;i < g.n;i++)
		for (j = 0;j < g.n;j++)
			g.edges[i][j] = A[i][j];
}
void DispMat(MatGraph g) {
	int i, j;
	for (i = 0;i < g.n;i++) {
		for (j = 0;j < g.n;j++) {
			if (g.edges[i][j] != INF)
				printf("%4d", g.edges[i][j]);
			else
				printf("%4s", "∞");
		}
		cout << endl;
	}
	
}
//邻接表
void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e) {
	int i, j = 0;
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
		cout << endl;;
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
int main() {
	MatGraph  g;
	AdjGraph* G;
	int A[MAXV][MAXV] = {
		{0,5,INF,7,INF,INF},{INF,0,4,INF,INF},
		{8,INF,0,INF,INF,9},{INF,INF,5,0,INF,6},
		{INF,INF,INF,5,0,INF},{3,INF,INF,INF,1,0}};
	int n = 6, e = 10;
	cout << "（1）邻接矩阵：" << endl;
	CreateMat(g, A, n, e);
	DispMat(g);
	cout << "（2）邻接表：" << endl;
	CreateAdj(G, A, n, e);
	DispAdj(G);
	cout << "（3）销毁邻接表：" << endl;
	DestroyAdj(G);
}
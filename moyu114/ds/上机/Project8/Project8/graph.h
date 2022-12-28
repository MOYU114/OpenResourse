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
	int n, e;
	VertexType vexs[MAXV];
}MatGraph;
//邻接矩阵
typedef struct ANode//声明边结点类型
{
	int adjvex;			//该边的终点编号
	struct ANode* nextarc;	//指向下一条边的指针
	int weight;		//该边的权值等信息
}  ArcNode;

typedef struct Vnode//声明邻接表头结点类型
{
	VertexType data;			//顶点信息
	//int count;                //存放入度，用于拓扑排序
	ArcNode* firstarc;		//指向第一条边
}  VNode;
typedef struct//声明图邻接表类型
{
	VNode adjlist[MAXV];	//邻接表
	int n, e;			//图中顶点数n和边数e
} AdjGraph;
//邻接矩阵
void CreateMat(MatGraph& g, int A[MAXV][MAXV], int n, int e);
void DispMat(MatGraph g);
//邻接表
void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e);
void DispAdj(AdjGraph* G);
void DestroyAdj(AdjGraph*& G);

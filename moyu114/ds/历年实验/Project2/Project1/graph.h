#include<iostream>
#include<malloc.h>
using namespace std;
#define MAXV 100
#define M 4
#define N 4
typedef char InfoType;
#define INF 32767
int maze[M + 2][N + 2] = {
	{1,1,1,1,1,1},{1,0,0,0,1,1},{1,0,1,0,0,1},
	{1,0,0,0,1,1},{1,1,0,0,0,1},{1,1,1,1,1,1} };
int visit[M + 2][N + 2] = { 0 };
typedef struct {
	int i;
	int j;
}Path;
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
	int i,j;			//该边的终点编号
	struct ANode* nextarc;	//指向下一条边的指针
}  ArcNode;

typedef struct Vnode//声明邻接表头结点类型
{
	ArcNode* firstarc;		//指向第一条边
}  VNode;
typedef struct//声明图邻接表类型
{
	VNode adjlist[M+2][N+2];	//邻接表
	int i, j;
} AdjGraph;

//邻接表
void CreateAdj(AdjGraph*& G);

//void DestroyAdj(AdjGraph*& G);

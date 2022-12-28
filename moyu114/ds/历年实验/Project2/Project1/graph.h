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
//�ڽӱ�
typedef struct //��������
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
//�ڽӾ���
typedef struct ANode//�����߽������
{
	int i,j;			//�ñߵ��յ���
	struct ANode* nextarc;	//ָ����һ���ߵ�ָ��
}  ArcNode;

typedef struct Vnode//�����ڽӱ�ͷ�������
{
	ArcNode* firstarc;		//ָ���һ����
}  VNode;
typedef struct//����ͼ�ڽӱ�����
{
	VNode adjlist[M+2][N+2];	//�ڽӱ�
	int i, j;
} AdjGraph;

//�ڽӱ�
void CreateAdj(AdjGraph*& G);

//void DestroyAdj(AdjGraph*& G);

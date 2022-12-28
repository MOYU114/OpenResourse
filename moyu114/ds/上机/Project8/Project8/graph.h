#include<iostream>
#include<malloc.h>
using namespace std;
#define MAXV 100
typedef char InfoType;
#define INF 32767
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
	int adjvex;			//�ñߵ��յ���
	struct ANode* nextarc;	//ָ����һ���ߵ�ָ��
	int weight;		//�ñߵ�Ȩֵ����Ϣ
}  ArcNode;

typedef struct Vnode//�����ڽӱ�ͷ�������
{
	VertexType data;			//������Ϣ
	//int count;                //�����ȣ�������������
	ArcNode* firstarc;		//ָ���һ����
}  VNode;
typedef struct//����ͼ�ڽӱ�����
{
	VNode adjlist[MAXV];	//�ڽӱ�
	int n, e;			//ͼ�ж�����n�ͱ���e
} AdjGraph;
//�ڽӾ���
void CreateMat(MatGraph& g, int A[MAXV][MAXV], int n, int e);
void DispMat(MatGraph g);
//�ڽӱ�
void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e);
void DispAdj(AdjGraph* G);
void DestroyAdj(AdjGraph*& G);

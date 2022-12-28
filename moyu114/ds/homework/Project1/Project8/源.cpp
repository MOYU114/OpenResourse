#include<iostream>
#include<malloc.h>
using namespace std;
#define MAXV 50
#define INF 32767
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

typedef struct Vnode
{
	VertexType data;			//������Ϣ
	ArcNode* firstarc;		//ָ���һ����
}  VNode;

typedef struct
{
	VNode adjlist[MAXV];	//�ڽӱ�
	int n, e;			//ͼ�ж�����n�ͱ���e
} AdjGraph;

void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e) {
	int i, j=0;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof AdjGraph);
	for (i = 0;i < n;i++) {
		G->adjlist[i].firstarc = NULL;//���ڽӱ�������ͷ����ָ�����ó�ֵ
	}
	for (i = 0;i < n;i++) {
		for (j = n - 1;j >= 0;j--) {
			if (A[i][j] != 0 && A[i][j] != INF) { //����һ����
				p = (ArcNode*)malloc(sizeof ArcNode);//����һ�����p
				p->adjvex = j;                     //����ڽӵ�
				p->weight = A[i][j];              //���Ȩ
				p->nextarc = G->adjlist[i].firstarc;//����ͷ�巨������p
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
		printf("��\n");
	}
}
void DestroyAdj(AdjGraph*& G) {
	int i;
	ArcNode* pre, * p;

	for (i = 0;i < G->n;i++) {//ɨ�����еĵ�����
		pre = G->adjlist[i].firstarc;//pָ���i����������׽��
		if (pre != NULL) {
			p = pre->nextarc;
			while (p != NULL)	//�ͷŵ�i������������б߽��
			{
				free(pre);
				pre = p; p = p->nextarc;
			}
			free(pre);
		}
	}
	free(G);//�ͷ�ͷ�������
}
void MatToList(MatGraph g, AdjGraph*& G) {
	//���ڽӾ���gת�����ڽӱ�G
	int i, j;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof(AdjGraph));
	for (i = 0;i < G->n;i++) {//���ڽӱ�������ͷ����ָ�����ó�ֵ
		G->adjlist[i].firstarc = NULL;
	}
	for (i = 0;i < g.n;i++) {//����ڽӾ�����ÿ��Ԫ��
		for (j = g.n - 1;j >= 0;j--) {
			if (g.edges[i][j] != 0 && g.edges[i][j] != INF) {//����һ����
				p = (ArcNode*)malloc(sizeof(ArcNode));//��һ���߽��p
				p->adjvex = j;p->weight = g.edges[i][j];
				p->nextarc = G->adjlist[i].firstarc;;     //����ͷ�巨������p
				G->adjlist[i].firstarc = p;
			}
		}
	}
	G->n = g.n;G->e = g.e;
}
void ListToMat(AdjGraph* G, MatGraph& g)
//���ڽӱ�Gת�����ڽӾ���g
{
	int i;
	ArcNode* p;
	for (i = 0;i < G->n;i++)			//ɨ�����еĵ�����
	{
		p = G->adjlist[i].firstarc;		//pָ���i����������׽��
		while (p != NULL)		//ɨ���i��������
		{
			g.edges[i][p->adjvex] = 1;//��vi���ܹ������·����һ
			p = p->nextarc;
		}
	}
	g.n = G->n; g.e = G->e;
}
int visited[MAXV];
void findpath(AdjGraph* G, int u, int v, int path[], int d, int length)
{ //d��ʾpath�ж����������ʼΪ0��length��ʾ·�����ȣ���ʼΪ0
	int w, i;
	ArcNode* p;
	path[d] = u; d++; //����u���뵽·���У�d��1
	visited[u] = 1; //���ѷ��ʱ��
	if (u == v && d > 0) //�ҵ�һ��·�������
	{
		printf(" ·������:%d, ·��:", length);
		for (i = 0;i < d;i++)
			printf("%2d", path[i]);
		printf("\n");
	}
	p = G->adjlist[u].firstarc; //pָ�򶥵�u�ĵ�һ���ڽӵ�
	while (p != NULL)
	{
		w = p->adjvex; //wΪ����u���ڽӵ�
		if (visited[w] == 0) //��w����δ����,�ݹ������
			findpath(G, w, v, path, d, p->weight + length);
		p = p->nextarc; //pָ�򶥵�u����һ���ڽӵ�
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
	CreateAdj(G, A, n, e); //�������̡̳���ͼ8.35���ڽӱ�
	printf("ͼG���ڽӱ�:\n");
	DispAdj(G); //����ڽӱ�
	int u = 0, v = 5;
	int path[MAXV];
	printf("��%d->%d������·��:\n", u, v);
	findpath(G, u, v, path, 0, 0);
	DestroyAdj(G);
	return 1;
}


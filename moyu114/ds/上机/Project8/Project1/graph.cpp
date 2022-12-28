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
	int n,e;
	VertexType vexs[MAXV];
}MatGraph;
//�ڽӾ���
typedef struct ANode//�����߽������
{
	int adjvex;			//�ñߵ��յ���
	struct ANode* nextarc;	//ָ����һ���ߵ�ָ��
	int weight;		//�ñߵ�Ȩֵ����Ϣ
}  ArcNode;

typedef struct Vnode//������ͷ�������
{
	VertexType data;			//������Ϣ

	ArcNode* firstarc;		//ָ���һ����
}  VNode;
typedef struct//����ͼ�ڽӱ�����
{
	VNode adjlist[MAXV];	//�ڽӱ�
	int n,e;			//ͼ�ж�����n�ͱ���e
} AdjGraph;
//�ڽӾ���
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
				printf("%4s", "��");
		}
		cout << endl;
	}
	
}
//�ڽӱ�
void CreateAdj(AdjGraph*& G, int A[MAXV][MAXV], int n, int e) {
	int i, j = 0;
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
		cout << endl;;
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
int main() {
	MatGraph  g;
	AdjGraph* G;
	int A[MAXV][MAXV] = {
		{0,5,INF,7,INF,INF},{INF,0,4,INF,INF},
		{8,INF,0,INF,INF,9},{INF,INF,5,0,INF,6},
		{INF,INF,INF,5,0,INF},{3,INF,INF,INF,1,0}};
	int n = 6, e = 10;
	cout << "��1���ڽӾ���" << endl;
	CreateMat(g, A, n, e);
	DispMat(g);
	cout << "��2���ڽӱ�" << endl;
	CreateAdj(G, A, n, e);
	DispAdj(G);
	cout << "��3�������ڽӱ�" << endl;
	DestroyAdj(G);
}
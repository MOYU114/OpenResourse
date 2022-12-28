
#include"graph.h"
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
	G = (AdjGraph*)malloc(sizeof (AdjGraph));
	for (i = 0;i < n;i++) {
		G->adjlist[i].firstarc = NULL;//���ڽӱ�������ͷ����ָ�����ó�ֵ
	}
	for (i = 0;i < n;i++) {
		for (j = n - 1;j >= 0;j--) {
			if (A[i][j] != 0 && A[i][j] != INF) { //����һ����
				p = (ArcNode*)malloc(sizeof (ArcNode));//����һ�����p
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


#include"graph.h"
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
	G = (AdjGraph*)malloc(sizeof (AdjGraph));
	for (i = 0;i < n;i++) {
		G->adjlist[i].firstarc = NULL;//给邻接表中所有头结点的指针域置初值
	}
	for (i = 0;i < n;i++) {
		for (j = n - 1;j >= 0;j--) {
			if (A[i][j] != 0 && A[i][j] != INF) { //存在一条边
				p = (ArcNode*)malloc(sizeof (ArcNode));//创建一个结点p
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

#include"graph.h"
void CreateAdj(AdjGraph*& G) {
	int i, j = 0;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof(AdjGraph));
	for (i = 0;i <= M;i++) {
		for (j = 0;j <= N;j++)
			G->adjlist[i][j].firstarc = NULL;//给邻接表中所有头结点的指针域置初值
	}
	p=(ArcNode*)malloc(sizeof(ArcNode));
	p->i = 1;p->j = 1;
	p->nextarc = NULL;
	G->adjlist[1][1].firstarc = p;
	for (i = 1;i <= M;i++) {
		for (j = 1;j <= N;j++)
			if (G->adjlist[i][j].firstarc!=NULL) {
				if (maze[i - 1][j] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i - 1;p->j = j;
					p->nextarc = G->adjlist[i - 1][j].firstarc;
					G->adjlist[i - 1][j].firstarc = p;
				}
				else if (maze[i][j-1] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i;p->j = j - 1;
					p->nextarc = G->adjlist[i][j - 1].firstarc;
					G->adjlist[i][j-1].firstarc = p;
				}
				else if (maze[i + 1][j] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i + 1;p->j = j;
					p->nextarc = G->adjlist[i + 1][j].firstarc;
					G->adjlist[i + 1][j].firstarc = p;
				}
				else if (maze[i][j+1] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i;p->j = j+1;
					p->nextarc = G->adjlist[i][j + 1].firstarc;
					G->adjlist[i][j + 1].firstarc = p;
				}
			}
	}
	G->i = M + 2;G->j = N + 2;
}

//void DestroyAdj(AdjGraph*& G) {
//	int i,j;
//	ArcNode* pre, * p;
//
//	for (i = 0;i < G->i;i++) {//扫描所有的单链表
//		for (j = 0;j < G->j;j++)
//		pre = G->adjlist[i][j].firstarc;//p指向第i个单链表的首结点
//		if (pre != NULL) {
//			p = pre->nextarc;
//			while (p != NULL)	//释放第i个单链表的所有边结点
//			{
//				free(pre);
//				pre = p; p = p->nextarc;
//			}
//			free(pre);
//		}
//	}
//	free(G);//释放头结点数组
//}
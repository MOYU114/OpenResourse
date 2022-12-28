#include"graph.h"
void CreateAdj(AdjGraph*& G) {
	int i, j = 0;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof(AdjGraph));
	for (i = 0;i <= M;i++) {
		for (j = 0;j <= N;j++)
			G->adjlist[i][j].firstarc = NULL;//���ڽӱ�������ͷ����ָ�����ó�ֵ
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
//	for (i = 0;i < G->i;i++) {//ɨ�����еĵ�����
//		for (j = 0;j < G->j;j++)
//		pre = G->adjlist[i][j].firstarc;//pָ���i����������׽��
//		if (pre != NULL) {
//			p = pre->nextarc;
//			while (p != NULL)	//�ͷŵ�i������������б߽��
//			{
//				free(pre);
//				pre = p; p = p->nextarc;
//			}
//			free(pre);
//		}
//	}
//	free(G);//�ͷ�ͷ�������
//}
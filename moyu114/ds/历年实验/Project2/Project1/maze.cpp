#include"queue.h"
#define MAXV 100
#define M 4
#define N 4
int maze[M + 2][N + 2] = {
	{1,1,1,1,1,1},{1,0,0,0,1,1},{1,0,1,0,0,1},
	{1,0,0,0,1,1},{1,1,0,0,0,1},{1,1,1,1,1,1} };
int visit[M + 2][N + 2] = { 0 };

typedef struct ANode//�����߽������
{
	int i, j;			//�ñߵ��յ���
	struct ANode* nextarc;	//ָ����һ���ߵ�ָ��
}  ArcNode;

typedef struct Vnode//�����ڽӱ�ͷ�������
{
	ArcNode* firstarc;		//ָ���һ����
}  VNode;
typedef struct//����ͼ�ڽӱ�����
{
	VNode adjlist[M + 2][N + 2];	//�ڽӱ�
	int i, j;
} AdjGraph;
typedef struct Node {
	int i, j;
	struct Node* next;
}Link;
void CreateAdj(AdjGraph*& G) {
	int i, j = 0;
	ArcNode* p;
	G = (AdjGraph*)malloc(sizeof(AdjGraph));
	for (i = 0;i <= M;i++) {
		for (j = 0;j <= N;j++)
			G->adjlist[i][j].firstarc = NULL;//���ڽӱ�������ͷ����ָ�����ó�ֵ
	}
	p = (ArcNode*)malloc(sizeof(ArcNode));
	p->i = 1;p->j = 1;
	p->nextarc = NULL;

	for (i = 1;i <= M;i++) {
		for (j = 1;j <= N;j++)
			if (maze[i][j]==0) {
				if (maze[i - 1][j] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i - 1;p->j = j;
					p->nextarc = G->adjlist[i][j].firstarc;
					G->adjlist[i][j].firstarc = p;
				}
				if (maze[i][j - 1] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i;p->j = j - 1;
					p->nextarc = G->adjlist[i][j].firstarc;
					G->adjlist[i][j].firstarc = p;
				}
				if (maze[i + 1][j] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i + 1;p->j = j;
					p->nextarc = G->adjlist[i][j].firstarc;
					G->adjlist[i][j].firstarc = p;
				}
				if (maze[i][j + 1] == 0) {
					p = (ArcNode*)malloc(sizeof(ArcNode));
					p->i = i;p->j = j + 1;
					p->nextarc = G->adjlist[i][j].firstarc;
					G->adjlist[i][j].firstarc = p;
				}
			}
	}
	G->i = M + 2;G->j = N + 2;
}

int visited[M + 2][N + 2] = { 0 };
bool BFS(AdjGraph* g, int i, int j) {
	ArcNode* p, * q;
	int i0, j0, i1, j1;
	/*Link* path, * p1, * p2;
	path = (Link*)malloc(sizeof(Link));
	p1 = path;
	p2 = (Link*)malloc(sizeof(Link));
	p1->next = p2;
	p2->i = i;	p2->j = j;
	p2->next = NULL;
	p1 = p2;*/
	SqQueue* qu,*pt;
	InitQueue(qu);
	InitQueue(pt);
	visited[i][j] = 2;
	enQueue(qu, i, j);
	enQueue(pt, i, j);
	while (!QueueEmpty(qu)) {
		deQueue(qu, i0, j0);

		p = g->adjlist[i0][j0].firstarc;
		while (p != NULL) {
			if (p->i == 4 && p->j == 4) {
				i0 = 1;j0 = 1;
				visited[p->i][p->j] = 1;
				while (i0 <= 4 && j0 <= 4) {
					int i2, j2;
					deQueue(pt, i2, j2);
					printf("[%d,%d] ", i2, j2);

				}
				return  true;
			}
			q = g->adjlist[i0][j0].firstarc;
			for (int di = 0;di < 4;di++)		//ѭ��ɨ��ÿ����λ
			{
				if (q != NULL)q = q->nextarc;
				else break;
				
				switch (di)
				{
				case 0:i1 = i0 - 1; j1 = j0;   break;
				case 1:i1 = i0;   j1 = j0 + 1; break;
				case 2:i1 = i0 + 1; j1 = j0;   break;
				case 3:i1 = i0;   j1 = j0 - 1; break;
				}
				if (visited[i1][j1] == 0) {//����
					enQueue(pt, i1, j1);
					enQueue(qu, i1, j1);	//(i1��j1)�������
					visited[i1][j1] = 1;//���丳ֵ1����ʾ���߹�
				}
				
			}

		}
		return false;
	}
}
int main() {
	AdjGraph* g;
	int visited[M + 2][N + 2] = { 0 };
	int cnt = 0;
	CreateAdj(g);
	cnt=BFS(g,1,1);
	/*for (int i = 0;i < M+2;i++) {
		for (int j = 0;j < N + 2;j++) {
			if(visited[i][j]==1)
				printf("(%d,%d)->",i,j);
		}*/
		
	}

